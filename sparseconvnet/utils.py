# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, glob, os, numpy as np, math
from .sparseConvNetTensor import SparseConvNetTensor
from .metadata import Metadata

def toLongTensor(dimension, x):
    if hasattr(x, 'type') and x.type() == 'torch.LongTensor':
        return x
    elif isinstance(x, (list, tuple)):
        assert len(x) == dimension
        return torch.LongTensor(x)
    else:
        return torch.LongTensor(dimension).fill_(x)


def optionalTensor(a, b):
    return getattr(a, b) if hasattr(a, b) else torch.Tensor()


def optionalTensorReturn(a):
    return a if a.numel() else None


def threadDatasetIterator(d):
    try:
        import queue
    except BaseException:
        import Queue as queue
    import threading

    def iterator():
        def worker(i):
            for k in range(i, len(d), 8):
                q.put(d[k])
        q = queue.Queue(16)
        for i in range(8):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
        for _ in range(len(d)):
            item = q.get()
            yield item
            q.task_done()
        q.join()
    return iterator



def concatenate_feature_planes(input):
    output = SparseConvNetTensor()
    output.metadata = input[0].metadata
    output.spatial_size = input[0].spatial_size
    output.features = torch.cat([i.features for i in input], 1)
    return output


def add_feature_planes(input):
    output = SparseConvNetTensor()
    output.metadata = input[0].metadata
    output.spatial_size = input[0].spatial_size
    output.features = sum([i.features for i in input])
    return output

def append_tensors(tensors):
    spatial_size=tensors[0].spatial_size
    dimension=len(spatial_size)
    x=SparseConvNetTensor(
        features=torch.cat([t.features for t in tensors],0),
        metadata=Metadata(dimension),
        spatial_size=spatial_size)
    for t in tensors:
        x.metadata.appendMetadata(t.metadata,spatial_size)
    return x

class AddCoords(torch.nn.Module):
    def forward(self, input):
        output = SparseConvNetTensor()
        if input.features.numel():
            with torch.no_grad():
                coords = input.get_spatial_locations()
                d = (input.spatial_size.type_as(input.features)-1)/2
                coords=coords[:,:-1].type_as(input.features)/ d[None,:] - 1
            output.features = torch.cat([input.features,coords],1)
        else:
            output.features = input.features
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output

def compare_sparse(x, y):
    cL,cR,L,R = x.metadata.compareSparseHelper(y.metadata, x.spatial_size)
    if x.features.is_cuda:
        cL=cL.cuda()
        cR=cR.cuda()
        L=L.cuda()
        R=R.cuda()
    e = 0
    if cL.numel():
        e += (x.features[cL]-y.features[cR]).pow(2).sum()
    if L.numel():
        e += x.features[L].pow(2).sum()
    if R.numel():
        e += y.features[R].pow(2).sum()
    return e / (cL.numel() + L.numel() + R.numel())

def spectral_norm_svd(module):
    w=module.weight
    if w.ndimension()==3:
        w=w.view(-1,w.size(2))
    _,s,_=torch.svd(w)
    return s[0]

def pad_with_batch_idx(x,idx): #add a batch index to the list of coordinates
    return torch.cat([x,torch.LongTensor(x.size(0),1).fill_(idx)],1)

def batch_location_tensors(location_tensors):
    a=[]
    for batch_idx, lt in enumerate(location_tensors):
        if lt.numel():
            a.append(pad_with_batch_idx(lt,batch_idx))
    return torch.cat(a,0)

def prepare_BLInput(batch):
    with torch.no_grad():
        n=max([l.size(0) for l,f in batch])
        l,f=batch[0]
        L=torch.empty(len(batch),n,l.size(1),dtype=torch.int64).fill_(-1)
        F=torch.zeros(len(batch),n,f.size(1))
        for i, (l, f) in enumerate(batch):
            L[i,:l.size(0),:].copy_(l)
            F[i,:f.size(0),:].copy_(f)
    return [L,F]

def checkpoint_restore(model,exp_name,name2,use_cuda=True,epoch=0):
    if use_cuda:
        model.cpu()
    if epoch>0:
        f=exp_name+'-%09d-'%epoch+name2+'.pth'
        assert os.path.isfile(f)
        print('Restore from ' + f)
        model.load_state_dict(torch.load(f))
    else:
        f=sorted(glob.glob(exp_name+'-*-'+name2+'.pth'))
        if len(f)>0:
            f=f[-1]
            print('Restore from ' + f)
            model.load_state_dict(torch.load(f))
            epoch=int(f[len(exp_name)+1:-len(name2)-5])
    if use_cuda:
        model.cuda()
    return epoch+1

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)

def is_square(num):
    return int(num**0.5+0.5)**2==num

def has_only_one_nonzero_digit(num): #https://oeis.org/A037124
    return num != 0 and (num/10**math.floor(math.log(num,10))).is_integer()

def checkpoint_save(model,exp_name,name2,epoch, use_cuda=True):
    f=exp_name+'-%09d-'%epoch+name2+'.pth'
    model.cpu()
    torch.save(model.state_dict(),f)
    if use_cuda:
        model.cuda()
    #remove previous checkpoints unless they are a power of 2 to save disk space
    epoch=epoch-1
    f=exp_name+'-%09d-'%epoch+name2+'.pth'
    if os.path.isfile(f):
        if not is_power2(epoch):
            os.remove(f)

def random_rotation(dimension=3,allow_mirror=False):
    r=torch.qr(torch.randn(dimension,dimension))[0]
    f=torch.randint(2,(3,))
    if f.sum()%2==0 and not allow_mirror:
        f=1-f
    return r*(2*f-1).float()

def squareroot_rotation(a):
    import scipy.spatial
    b=scipy.spatial.transform.Slerp(
        [0,1],
        scipy.spatial.transform.Rotation.from_dcm(torch.stack([torch.eye(3),a])))([0.5]).as_dcm()
    return torch.from_numpy(b).float()[0]

def voxelize_pointcloud(xyz,rgb,average=True,accumulate=False):
    if xyz.numel()==0:
        return xyz, rgb
    if average or accumulate:
        xyz,inv,counts=np.unique(xyz.numpy(),axis=0,return_inverse=True,return_counts=True)
        xyz=torch.from_numpy(xyz)
        inv=torch.from_numpy(inv)
        rgb_out=torch.zeros(xyz.size(0),rgb.size(1),dtype=torch.float32)
        rgb_out.index_add_(0,inv,rgb)
        if average:
            rgb=rgb_out/torch.from_numpy(counts[:,None]).float()
        return xyz, rgb
    else:
        xyz,idxs=np.unique(xyz,axis=0,return_index=True)
        xyz=torch.from_numpy(xyz)
        rgb=rgb[idxs]
        return xyz, rgb

class checkpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, x_features, x_metadata, x_spatial_size):
        ctx.run_function = run_function
        ctx.save_for_backward(x_features, x_spatial_size)
        ctx.x_metadata=x_metadata
        with torch.no_grad():
            y = run_function(
                SparseConvNetTensor
                (x_features, x_metadata, x_spatial_size))
        return y.features
    @staticmethod
    def backward(ctx, grad_y_features):
        x_features, x_spatial_size = ctx.saved_tensors
        x_features = x_features.detach()
        x_features.requires_grad = True
        with torch.enable_grad():
            y = ctx.run_function(
                SparseConvNetTensor
                (x_features, ctx.x_metadata, x_spatial_size))
        torch.autograd.backward(y.features, grad_y_features,retain_graph=False)
        return None, x_features.grad, None, None

def checkpoint101(run_function, x, down=1):
    f=checkpointFunction.apply(run_function, x.features, x.metadata, x.spatial_size)
    s=x.spatial_size//down
    return SparseConvNetTensor(f, x.metadata, s)

def matplotlib_cubes(ax, positions,colors):
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    ...
    plt.show()
    """
    try:
        positions=positions.numpy()
        colors=colors.numpy()
        X = np.array([[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
             [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
             [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
             [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
             [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
             [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]).astype(np.float32)[None]-0.5
        X=X+positions[:,None,None,:]
        X.resize(X.shape[0]*6,4,3)
        m=positions.min(0)
        M=positions.max(0)+1
        ax.set_xlim([m[0],M[0]])
        ax.set_ylim([m[1],M[1]])
        ax.set_zlim([m[2],M[2]])
        ax.add_collection3d(Poly3DCollection(X,
                                facecolors=np.repeat(colors,6, axis=0)))
    except:
        print('matplotlibcubes fail!?!')
        pass
    ax.set_axis_off()
def matplotlib_planes(ax, positions,colors):
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    ...
    plt.show()
    """
    try:
        positions=positions.numpy()
        colors=colors.numpy()
        X = np.array([[[0, -0.5, 0.5], [0, -0.5, -0.5], [0, 0.5, -0.5], [0, 0.5, 0.5]]]).astype(np.float32)[None]
        X=X+positions[:,None,None,:]
        X.resize(X.shape[0]*1,4,3)
        m=positions.min(0)
        M=positions.max(0)+1
        ax.set_xlim([m[0],M[0]])
        ax.set_ylim([m[1],M[1]])
        ax.set_zlim([m[2],M[2]])
        ax.add_collection3d(Poly3DCollection(X,
                                facecolors=np.repeat(colors,1, axis=0)))
    except:
        pass
    ax.set_axis_off()

def visdom_scatter(vis, xyz, rgb, win='3d', markersize=3, title=''):
    rgb=rgb.detach()
    rgb-=rgb.min()
    rgb/=rgb.max()/255+1e-10
    rgb=rgb.floor().cpu().numpy()
    vis.scatter(
        xyz.detach().cpu().numpy(),
        opts={'markersize': markersize,'markercolor': rgb, 'title': title},
        win=win)
    
def ply_scatter(name, xyz, rgb):
    rgb=rgb.detach()
    rgb-=rgb.min()
    rgb/=rgb.max()/255+1e-10
    rgb=rgb.floor().cpu().numpy()
    with open(name+'.ply','w') as f:
        print("""ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header"""%(xyz.size(0)), file = f)
        for (x,y,z),(r,g,b) in zip(xyz,rgb):
            print('%d %d %d %d %d %d'%(x,y,z,r,g,b),file=f)


class VerboseIdentity(torch.nn.Module):
    def forward(self, x):
        print(x)
        return x
