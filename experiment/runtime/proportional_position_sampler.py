"""Spatial coverage sampler; counts are provided by the proportional rule."""
import numpy as np

def stratified_positions(points,n):
    """Recursively divide the regular legal grid into equal-population strata.

    Each position represents the same physical grid-cell area. Splitting the
    longest axis in proportion to the requested number of leaves gives spatial
    coverage with nearly equal legal area; no errors or validation labels used.
    """
    assert len(points)>=n>0
    leaves=[]
    def divide(ids,k):
        if k==1:
            center=points[ids].astype(np.float64).mean(axis=0)
            distance=((points[ids]-center)**2).sum(axis=1)
            chosen=int(ids[int(np.argmin(distance))])
            leaves.append(dict(index=len(leaves),position_index=chosen,x=float(points[chosen,0]),
                y=float(points[chosen,1]),grid_positions_in_stratum=len(ids),
                minimum=points[ids].min(axis=0).tolist(),maximum=points[ids].max(axis=0).tolist()))
            return
        axis=int(np.argmax(np.ptp(points[ids],axis=0)))
        order=np.lexsort((points[ids,1-axis],points[ids,axis]));ids=ids[order]
        left=k//2;cut=int(round(len(ids)*left/k));cut=max(left,min(len(ids)-(k-left),cut))
        divide(ids[:cut],left);divide(ids[cut:],k-left)
    divide(np.arange(len(points)),n)
    chosen=np.asarray([r['position_index'] for r in leaves],dtype=np.int64)
    assert len(np.unique(chosen))==n and sum(r['grid_positions_in_stratum'] for r in leaves)==len(points)
    return chosen,leaves
