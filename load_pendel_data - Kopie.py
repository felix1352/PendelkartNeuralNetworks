import h5py
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def LoadHdf5Mat(matfilePath):
    def UnpackHdf5Group(hdf5Matfile, hdf5Group):
        assert type(hdf5Group) == h5py._hl.group.Group
        return {key: UnpackHdf5(hdf5Matfile, hdf5Group[key]) for key in hdf5Group}

    def UnpackHdf5Dataset(hdf5Matfile, hdf5Group):
        assert type(hdf5Group) == h5py._hl.dataset.Dataset

        if hdf5Group.dtype == np.dtype("object"):
            out = np.ndarray(hdf5Group.shape, dtype=object)
            with np.nditer(
                    [out, hdf5Group],
                    flags=["refs_ok"],
                    op_flags=[["writeonly"], ["readonly"]],
            ) as it:
                for valOut, valIn in it:
                    if type(valIn[()]) in (
                            h5py._hl.group.Group,
                            h5py._hl.dataset.Dataset,
                            h5py.h5r.Reference,
                    ):
                        valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
                    else:
                        valOut[()] = valIn[()]

                return it.operands[0].squeeze()

        elif hdf5Group.dtype.kind in ("u", "i", "f"):
            # Deal with empty arrays
            if "MATLAB_empty" in hdf5Group.attrs.keys():
                return np.ndarray(0)

            # Value is encoded as int but needs to be decoded
            if "MATLAB_int_decode" in hdf5Group.attrs.keys():
                if "MATLAB_class" not in hdf5Group.attrs.keys():
                    raise NotImplementedError("MATLAB_int_decode but no MATLAB_class")

                if hdf5Group.attrs["MATLAB_class"] == b"char":
                    return hdf5Group[()].tobytes().decode("utf16")

                if hdf5Group.attrs["MATLAB_class"] == b"logical":
                    return hdf5Group[()].squeeze()

                raise NotImplementedError(
                    "MATLAB_int_decode but unsupported MATLAB_class"
                )

            else:
                return hdf5Group[()].squeeze()

        else:
            raise NotImplementedError(
                f"Decode not implemented for dtype {hdf5Group.dtype}"
            )

    def UnpackHdf5(hdf5Matfile, hdf5Group):
        if type(hdf5Group) is h5py._hl.group.Group:
            return UnpackHdf5Group(hdf5Matfile, hdf5Group)

        elif type(hdf5Group) is h5py._hl.dataset.Dataset:
            return UnpackHdf5Dataset(hdf5Matfile, hdf5Group)

        elif type(hdf5Group) is h5py.h5r.Reference:
            return UnpackHdf5(hdf5Matfile, hdf5Matfile[hdf5Group])

    hdf5Matfile = h5py.File(matfilePath, "r")

    out = {}
    for key in hdf5Matfile:
        # Skip the #refs# entry
        if key == "#refs#":
            continue

        out[key] = UnpackHdf5(hdf5Matfile, hdf5Matfile[key])

    return next(iter(out.values()))


#data_upswing = LoadHdf5Mat('./aufschwung_regelung.mat')

path = Path('verlade_bruecke.mat')
data_upswing = LoadHdf5Mat(path)
time = data_upswing.get('X').get('Data')
data_names = data_upswing.get('Y').get('Name')
force = data_upswing.get('Y').get('Data')[0]
pos = data_upswing.get('Y').get('Data')[1]
angle = data_upswing.get('Y').get('Data')[3]

fig, ax = plt.subplots(3, sharex=True)
ax[0].plot(time, force)
ax[0].grid()
ax[0].set_ylabel('force')
ax[1].plot(time, pos)
ax[1].grid()
ax[1].set_ylabel('cart position')
ax[2].grid()
ax[2].set_ylabel('angle')
ax[2].axhline(linewidth=1, y=180, color='red')
ax[2].axhline(linewidth=1, y=0, color='red')
ax[2].plot(time, angle)
ax[2].text(40, 180, 'untere Ruhelage', fontsize=6, va='center', ha='center', backgroundcolor='w')
ax[2].text(110, 0, 'obere Ruhelage', fontsize=6, va='center', ha='center', backgroundcolor='w')
ax[2].set_ylim([-50, 330])
ax[2].set_xlabel('time')

#plt.show()
plt.show(block=False)