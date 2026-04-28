#### creates a movie of the time evolution of multifrequency radio contours overlaid on AIA, with Stokes I and V spectra indicating the time and frequency
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.lines import Line2D
from datetime import datetime
import astropy.units as u
from astropy.io import fits
import sunpy.map
import glob, os, subprocess
from astropy.coordinates import SkyCoord

from nenufar_module import process_multiple_maps, get_data, plot_contours

# =========================================================
# CONFIG
# =========================================================
# =========================================================
# CONFIG
# =========================================================
root = '/data/sbhunia/type_II_2024/'

SB_CONFIG = {
    "SB313": {
        "folder": "SB313_61_1/step_iocorrect_outputs_20240713/SB313/corr_fits",
        "start": 40,
        "pattern": "{sb}-t{tid:04d}-image_corrWCS.fits"
    },
    "SB316": {
        "folder": "SB316_61_7/step_iocorrect_outputs_20240713/SB316/corr_fits",
        "start": 40,
        "pattern": "{sb}-t{tid:04d}-image_corrWCS.fits"
    },
    "SB317": {
        "folder": "SB317_61_9/step_iocorrect_outputs_20240713/SB317/corr_fits",
        "start": 40,
        "pattern": "{sb}-t{tid:04d}-image_corrWCS.fits"
    },
    "SB320": {
        "folder": "SB320_62_49/step_iocorrect_outputs_20240713/SB320/corr_fits",
        "start": 40,
        "pattern": "{sb}-t{tid:04d}-image_corrWCS.fits"
    },
    "SB323": {
        "folder": "SB323_63/step_iocorrect_outputs_20240713/SB323/corr_fits",
        "start": 40,
        "pattern": "{sb}-t{tid:04d}-image_corrWCS.fits"
    }
}

sbs = list(SB_CONFIG.keys())

output_dir = "/data/sbhunia/type_II_2024/frame_outputs/"
os.makedirs(output_dir, exist_ok=True)

N_FILES = 217-23
collist = plt.cm.Blues(np.linspace(0, 1, len(sbs)+1))
color_map = dict(zip(sbs, collist))
# =========================================================
# RADIO SPECTRA
# =========================================================
def load_npz(filename):
    d = np.load(filename)
    data = np.transpose(d['data'])[0]
    freq = d['freq']
    times = [datetime.fromtimestamp(t) for t in d['time']]
    return data, freq, times

data_I, freq, times = load_npz(
    '/home/sbhunia/casa_flux_calibr/data_other_typeii/sun1_I.npz'
)

data_V, _, _ = load_npz(
    '/home/sbhunia/casa_flux_calibr/data_other_typeii/sun1_V.npz'
)

scale = np.load('/home/sbhunia/casa_flux_calibr/data_other_typeii/scale.npy')

dataI = data_I * scale[:, np.newaxis] / 1e4
dataV = data_V * scale[:, np.newaxis] / 1e4

# =========================================================
# AIA TIMES
# =========================================================
aia_list = sorted(glob.glob('/home/sbhunia/AIA_data/193/*.fits'))

aia_t = np.array([
    datetime.strptime(fits.open(f)[1].header['DATE-OBS'][:19],
                      '%Y-%m-%dT%H:%M:%S')
    for f in aia_list
])

# =========================================================
# MAIN LOOP
# =========================================================
for i in range(N_FILES):

    print(f"\nProcessing frame {i}")

    # -----------------------------------------------------
    # BUILD FITS LIST (CLEAN + SCALABLE)
    # -----------------------------------------------------
    fits_list = []
    sb_labels = []

    for sb, cfg in SB_CONFIG.items():

        tid = cfg["start"] + i
        fname = cfg["pattern"].format(sb=sb, tid=tid)

        fits_file = os.path.join(root, cfg["folder"], fname)

        fits_list.append(fits_file)
        sb_labels.append(sb)

    # -----------------------------------------------------
    # LOAD RADIO MAPS
    # -----------------------------------------------------
    maps = process_multiple_maps(fits_list)
    maps = sorted(maps, key=lambda x: x[1])  # sort by frequency

    m_ref, _, obstime = maps[0]
    target_time = obstime.to_datetime()
    target = dates.date2num(target_time)

    # -----------------------------------------------------
    # MATCH AIA
    # -----------------------------------------------------
    tidx = np.argmin(np.abs(aia_t - target_time))
    aia_map = sunpy.map.Map(aia_list[tidx])

    # =====================================================
    # FIGURE
    # =====================================================
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # -----------------------------------------------------
    # SPECTRA I
    # -----------------------------------------------------
    ax1.imshow(
        dataI,
        cmap='plasma',
        vmin=0.01, vmax=20,
        aspect='auto',
        origin='lower',
        extent=(times[0], times[-1], freq[0], freq[-1])
    )

    # -----------------------------------------------------
    # SPECTRA V
    # -----------------------------------------------------
    ax2.imshow(
        dataV,
        cmap='seismic',
        vmin=-1, vmax=1,
        aspect='auto',
        origin='lower',
        extent=(times[0], times[-1], freq[0], freq[-1])
    )

    ax1.scatter([target]*5, [61.13,61.71,61.91,62.49,63.08], color='white', s=15)
    ax2.scatter([target]*5, [61.13,61.71,61.91,62.49,63.08], color='black', s=15)

    ax1.set_title(f"Stokes I | {target_time.strftime('%H:%M:%S')}")
    ax2.set_title("Stokes V")

    # time formatting
    startt = datetime(2024, 7, 13, 12, 43, 29)
    endt = datetime(2024, 7, 13, 12, 46, 53)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(dates.MinuteLocator(interval=2))
        ax.xaxis.set_minor_locator(dates.SecondLocator(interval=30))
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
        ax.set_ylim(59, 64)
        ax.set_xlim(startt, endt)

    # -----------------------------------------------------
    # AIA PANEL
    # -----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :], projection=aia_map)
    aia_map.plot(axes=ax3, clip_interval=(1, 99.9) * u.percent)

    legend_handles = []

    # -----------------------------------------------------
    # MULTI-FREQUENCY OVERLAY
    # -----------------------------------------------------
    print("Plotting contours...")

    for (m, fval, _), sb in zip(maps, sb_labels):


        plot_contours(aia_map, m, ax3, color=color_map[sb])

        # peak emission
        y, x = np.unravel_index(np.nanargmax(m.data), m.data.shape)
        coord = m.pixel_to_world(x * u.pix, y * u.pix)

        ax3.plot(
            coord.Tx.to('deg'),
            coord.Ty.to('deg'),
            'o',
            color=color_map[sb],
            transform=ax3.get_transform('world'),
            ms=6
        )

        legend_handles.append(
            Line2D([0], [0], color=color_map[sb], lw=2,
                   label=f"{fval:.1f}")
        )

    ax3.legend(handles=legend_handles, loc='lower right')
    ax3.set_title(target_time.strftime('%H:%M:%S'))
    ax3.patch.set_facecolor('black')
    # -----------------------------------------------------
    # CROP AIA VIEW
    # -----------------------------------------------------
    xlims_world = [0, 1800] * u.arcsec
    ylims_world = [-1000, 570] * u.arcsec

    world_coords = SkyCoord(
        Tx=xlims_world,
        Ty=ylims_world,
        frame=aia_map.coordinate_frame
    )

    pixel_coords = aia_map.world_to_pixel(world_coords)

    ax3.set_xlim(pixel_coords.x.value)
    ax3.set_ylim(pixel_coords.y.value)

    # -----------------------------------------------------
    # SAVE FRAME
    # -----------------------------------------------------
    outname = os.path.join(output_dir, f"frame_{i:04d}.png")

    plt.tight_layout()
    #plt.show()
    plt.savefig(outname, dpi=200)
    plt.close(fig)

    print("Saved:", outname)

# =========================================================
# CREATE MOVIE
# =========================================================
print("\nCreating movie...")

cmd = [
    "ffmpeg", "-y",
    "-framerate", "24",
    "-i", os.path.join(output_dir, "frame_%04d.png"),
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    os.path.join(output_dir, "output.mp4")
]

subprocess.run(cmd)

print("Movie created!")

