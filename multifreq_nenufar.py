#### This code calls another script to make nenuFAR maps and plot images with multiple frequencies

from nenufar_module import process_multiple_maps
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import pdb

root = '/data/sbhunia/type_II_2024/'

fits_files = [
    root + 'SB313_61_1/step_iocorrect_outputs_20240713/SB313/corr_fits/SB313-t0017-image_corrWCS.fits',
    root + 'SB316_61_7/step_iocorrect_outputs_20240713/SB316/corr_fits/SB316-t0017-image_corrWCS.fits',
    root + 'SB317_61_9/step_iocorrect_outputs_20240713/SB317/corr_fits/SB317-t0017-image_corrWCS.fits',
    root + 'SB320_62_49/step_iocorrect_outputs_20240713/SB320/corr_fits/SB320-t0017-image_corrWCS.fits',
    root + 'SB323_63/step_iocorrect_outputs_20240713/SB323/corr_fits/SB323-t0017-image_corrWCS.fits']

maps = process_multiple_maps(fits_files)

fig = plt.figure(figsize=(15, 5))

for i, (m, freq, obstime) in enumerate(maps):
    ax = fig.add_subplot(1, len(fits_files), i+1, projection=m)
    #pdb.set_trace()
    m.plot(axes=ax, cmap="viridis")
    m.draw_limb(axes=ax)
    m.draw_grid(axes=ax)

    ax.set_title(f"{freq:.1f}")
    # ---- crop region ----
    xlims_world = [-3000, 3000] * u.arcsec
    ylims_world = [-3000, 3000] * u.arcsec

    world_coords = SkyCoord(
        Tx=xlims_world,
        Ty=ylims_world,
        frame=m.coordinate_frame)

    pixel_coords = m.world_to_pixel(world_coords)
    ax.set_xlim(pixel_coords.x.value)
    ax.set_ylim(pixel_coords.y.value)

plt.tight_layout()
plt.show()
