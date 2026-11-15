"""NE111 Term Project - Plot Digitizer (Streamlit single-page app)

Workflow
1. Upload an image of a graph.
2. Click 3 calibration points: origin, a point on the x-axis, a point on the y-axis,
   and enter the real axis values of those points in the sidebar.
3. Click data points. Each dataset has its own colour.
4. Edit the live table if needed, then download everything as a CSV.

Calibration uses an affine mapping built from the three clicked points, so scans
that are rotated (or slightly skewed) are handled automatically. An optional
rotation slider is also provided to straighten the image visually.
"""

import csv
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Plot Digitizer", page_icon="📈", layout="wide")

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#008080", "#f032e6", "#9a6324", "#808000", "#000075",
]
CAL_LABELS = ["Origin point", "Point on the x-axis", "Point on the y-axis"]
CAL_COLOR = "#ff00ff"
DISPLAY_W = 800
MAX_SIDE = 6000

ss = st.session_state


# --------------------------------------------------------------------------
# State helpers
# --------------------------------------------------------------------------
def init_state():
    ss.setdefault("datasets", [])
    ss.setdefault("next_id", 1)
    ss.setdefault("cal", [])
    ss.setdefault("last_click", None)
    ss.setdefault("img_id", None)
    ss.setdefault("base_img", None)
    ss.setdefault("angle", 0.0)
    ss.setdefault("shown_img", None)
    ss.setdefault("version", 0)
    ss.setdefault("new_name", "")


def add_dataset(name=None):
    i = ss.next_id
    ss.next_id += 1
    name = (name or "").strip() or f"Dataset {i}"
    ss.datasets.append(
        {"id": i, "name": name, "color": PALETTE[(i - 1) % len(PALETTE)], "points": []}
    )
    ss.active_id = i
    ss.version += 1


def cb_add_dataset():
    add_dataset(ss.get("new_name", ""))
    ss.new_name = ""


def cb_delete_dataset(ds_id):
    ss.datasets = [d for d in ss.datasets if d["id"] != ds_id]
    ss.pop("active_id", None)
    ss.version += 1


def cb_undo_point(ds_id):
    for d in ss.datasets:
        if d["id"] == ds_id and d["points"]:
            d["points"].pop()
    ss.version += 1


def cb_clear_points(ds_id):
    for d in ss.datasets:
        if d["id"] == ds_id:
            d["points"] = []
    ss.version += 1


def cb_undo_cal():
    if ss.cal:
        ss.cal.pop()
    ss.version += 1


def cb_reset_cal():
    ss.cal = []
    ss.version += 1


def reset_geometry():
    """Called when the image or its rotation changes: old pixel positions are invalid."""
    ss.cal = []
    for d in ss.datasets:
        d["points"] = []
    ss.version += 1


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def load_image(uploaded):
    """Load png/jpg/gif/tiff (colour, greyscale, palette, 16-bit...) as an RGB image."""
    img = Image.open(uploaded)
    try:
        img.seek(0)  # first frame of GIF / multi-page TIFF
    except Exception:
        pass
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=float)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        arr = (arr - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype("uint8"))
    if img.mode in ("RGBA", "LA", "P", "PA"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        f = MAX_SIDE / max(img.size)
        img = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)
    return img


def get_font(size=16):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# --------------------------------------------------------------------------
# Calibration maths (affine map from pixels to axis values)
# --------------------------------------------------------------------------
def parse_float(s):
    try:
        v = float(str(s).strip())
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def build_calibration(log_x, log_y, vals):
    """Return (cal, error). cal is None until 3 points are clicked and values valid."""
    if len(ss.cal) < 3:
        return None, None
    nums = [parse_float(v) for v in vals]
    if any(n is None for n in nums):
        return None, "Enter numeric values for all four calibration fields."
    x0, y0, x1, y1 = nums
    if x0 == x1 or y0 == y1:
        return None, "Calibration values for the two points on an axis must differ."
    if log_x and (x0 <= 0 or x1 <= 0):
        return None, "A logarithmic x-axis needs positive calibration values."
    if log_y and (y0 <= 0 or y1 <= 0):
        return None, "A logarithmic y-axis needs positive calibration values."

    O = np.array(ss.cal[0], dtype=float)
    u = np.array(ss.cal[1], dtype=float) - O
    v = np.array(ss.cal[2], dtype=float) - O
    det = u[0] * v[1] - v[0] * u[1]
    if abs(det) < 1e-9:
        return None, "The three calibration points are collinear. Please redo the calibration."

    fx = (lambda t: math.log10(t)) if log_x else (lambda t: t)
    fy = (lambda t: math.log10(t)) if log_y else (lambda t: t)
    return {
        "O": O, "u": u, "v": v, "det": det,
        "log_x": log_x, "log_y": log_y,
        "fx0": fx(x0), "fx1": fx(x1), "fy0": fy(y0), "fy1": fy(y1),
    }, None


def pixel_to_data(p, cal):
    d = np.array(p, dtype=float) - cal["O"]
    u, v = cal["u"], cal["v"]
    a = (d[0] * v[1] - v[0] * d[1]) / cal["det"]
    b = (u[0] * d[1] - u[1] * d[0]) / cal["det"]
    fx = cal["fx0"] + a * (cal["fx1"] - cal["fx0"])
    fy = cal["fy0"] + b * (cal["fy1"] - cal["fy0"])
    try:
        x = 10 ** fx if cal["log_x"] else fx
        y = 10 ** fy if cal["log_y"] else fy
    except OverflowError:
        return float("nan"), float("nan")
    return float(x), float(y)


def data_to_pixel(x, y, cal):
    """Inverse mapping, used when the user edits the table. Returns None if impossible."""
    try:
        if cal["log_x"]:
            if x <= 0:
                return None
            x = math.log10(x)
        if cal["log_y"]:
            if y <= 0:
                return None
            y = math.log10(y)
        a = (x - cal["fx0"]) / (cal["fx1"] - cal["fx0"])
        b = (y - cal["fy0"]) / (cal["fy1"] - cal["fy0"])
    except (ValueError, ZeroDivisionError):
        return None
    p = cal["O"] + a * cal["u"] + b * cal["v"]
    return (float(p[0]), float(p[1]))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_view(img, box, datasets):
    """Crop to the zoom window, scale to DISPLAY_W and draw all markers."""
    l, t, r, b = box
    cw, ch = r - l, b - t
    s = DISPLAY_W / cw
    disp = img.crop(box).resize((DISPLAY_W, max(1, int(round(ch * s)))), Image.LANCZOS)
    draw = ImageDraw.Draw(disp)
    font = get_font(16)
    rad = 5

    for d in datasets:
        for (px, py) in d["points"]:
            x, y = (px - l) * s, (py - t) * s
            if -rad <= x <= DISPLAY_W + rad and -rad <= y <= disp.height + rad:
                draw.ellipse([x - rad, y - rad, x + rad, y + rad],
                             fill=d["color"], outline="white", width=1)

    for i, (px, py) in enumerate(ss.cal):
        x, y = (px - l) * s, (py - t) * s
        if -20 <= x <= DISPLAY_W + 20 and -20 <= y <= disp.height + 20:
            draw.line([x - 9, y, x + 9, y], fill=CAL_COLOR, width=2)
            draw.line([x, y - 9, x, y + 9], fill=CAL_COLOR, width=2)
            draw.text((x + 8, y + 4), ["O", "X", "Y"][i], fill=CAL_COLOR, font=font)
    return disp


def build_csv(datasets, cal, x_label, y_label):
    cols = []
    for d in datasets:
        cols.append([pixel_to_data(p, cal) for p in d["points"]])
    n = max((len(c) for c in cols), default=0)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([v for d in datasets for v in (d["name"], "")])
    w.writerow([v for _ in datasets for v in (x_label, y_label)])
    for i in range(n):
        row = []
        for c in cols:
            if i < len(c):
                row += [f"{c[i][0]:.10g}", f"{c[i][1]:.10g}"]
            else:
                row += ["", ""]
        w.writerow(row)
    return buf.getvalue()


def dataset_table(ds, cal, x_label, y_label):
    """Editable table for one dataset. Edits are converted back to pixel positions."""
    vals = [pixel_to_data(p, cal) for p in ds["points"]]
    df = pd.DataFrame(vals, columns=["x", "y"], dtype=float)
    edited = st.data_editor(
        df,
        key=f"editor_{ds['id']}_{ss.version}",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "x": st.column_config.NumberColumn(x_label, format="%.6g"),
            "y": st.column_config.NumberColumn(y_label, format="%.6g"),
        },
    )
    same = list(edited.index) == list(df.index) and np.allclose(
        edited.to_numpy(dtype=float), df.to_numpy(dtype=float),
        rtol=1e-9, atol=0, equal_nan=True)
    if same:
        return

    old = dict(zip(df.index, ds["points"]))
    new_pts, skipped = [], 0
    for idx, row in edited.iterrows():
        x, y = row["x"], row["y"]
        if pd.isna(x) or pd.isna(y):
            continue
        if idx in old:
            ox, oy = df.loc[idx, "x"], df.loc[idx, "y"]
            if math.isclose(x, ox, rel_tol=1e-9, abs_tol=0) and math.isclose(y, oy, rel_tol=1e-9, abs_tol=0):
                new_pts.append(old[idx])
                continue
        p = data_to_pixel(float(x), float(y), cal)
        if p is None:
            skipped += 1
        else:
            new_pts.append(p)
    ds["points"] = new_pts
    ss.version += 1
    if skipped:
        st.toast(f"{skipped} row(s) skipped (invalid value for a log axis).")
    st.rerun()


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
init_state()
st.title("📈 Plot Digitizer")
st.caption("Extract x-y data from an image of a graph.")

uploaded = st.file_uploader(
    "Upload an image of a graph",
    type=["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp"],
)

if uploaded is None:
    st.info("Upload a graph image to get started (PNG, JPG, GIF or TIFF; colour or greyscale).")
    st.stop()

img_id = f"{uploaded.name}-{uploaded.size}"
if ss.img_id != img_id:
    try:
        ss.base_img = load_image(uploaded)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read this image: {e}")
        st.stop()
    ss.img_id = img_id
    ss.angle = 0.0
    ss.shown_img = ss.base_img
    ss.last_click = None
    ss.datasets = []
    ss.next_id = 1
    ss.cal = []
    ss.pop("active_id", None)
    add_dataset()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("1. Axes")
    x_label = st.text_input("x-axis label", value="x", key="x_label")
    y_label = st.text_input("y-axis label", value="y", key="y_label")
    c1, c2 = st.columns(2)
    log_x = c1.checkbox("Log x-axis", key="log_x")
    log_y = c2.checkbox("Log y-axis", key="log_y")

    st.header("2. Calibration")
    st.caption(
        "Click three points on the image, then enter their real values below. "
        "The *origin* can be any point where you know both x and y "
        "(for a log axis it cannot be 0). The x-axis point should lie along the "
        "x-axis and the y-axis point along the y-axis."
    )
    cx0 = st.text_input("Origin point: x value", "0", key="cal_x0")
    cy0 = st.text_input("Origin point: y value", "0", key="cal_y0")
    cx1 = st.text_input("x-axis point: x value", "1", key="cal_x1")
    cy1 = st.text_input("y-axis point: y value", "1", key="cal_y1")
    b1, b2 = st.columns(2)
    b1.button("Undo cal. point", on_click=cb_undo_cal, disabled=not ss.cal, use_container_width=True)
    b2.button("Reset calibration", on_click=cb_reset_cal, disabled=not ss.cal, use_container_width=True)

    st.header("3. Image rotation")
    angle = st.slider(
        "Straighten scan (degrees, counter-clockwise)", -45.0, 45.0,
        value=0.0, step=0.1, key="angle_slider",
        help="Optional. The calibration already corrects for rotation, "
             "but straightening makes clicking easier. Changing this clears points.",
    )
    if angle != ss.angle:
        ss.angle = angle
        ss.shown_img = ss.base_img.rotate(
            angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)
        ) if angle else ss.base_img
        reset_geometry()
        st.rerun()

    st.header("4. Datasets")
    ids = [d["id"] for d in ss.datasets]
    if ss.get("active_id") not in ids:
        ss.active_id = ids[0] if ids else None

    active = None
    if ids:
        names = {d["id"]: d["name"] for d in ss.datasets}
        st.selectbox("Active dataset (receives clicks)", ids,
                     format_func=lambda i: names[i], key="active_id")
        active = next(d for d in ss.datasets if d["id"] == ss.active_id)
        new_name = st.text_input("Rename active dataset", value=active["name"],
                                 key=f"rename_{active['id']}")
        active["name"] = new_name.strip() or active["name"]
        st.markdown(
            f"<span style='color:{active['color']};font-size:22px'>●</span> "
            f"points: **{len(active['points'])}**", unsafe_allow_html=True)
        u1, u2 = st.columns(2)
        u1.button("Undo last point", on_click=cb_undo_point, args=(active["id"],),
                  disabled=not active["points"], use_container_width=True)
        u2.button("Clear points", on_click=cb_clear_points, args=(active["id"],),
                  disabled=not active["points"], use_container_width=True)
        st.button("🗑️ Delete dataset", on_click=cb_delete_dataset, args=(active["id"],),
                  use_container_width=True)
    else:
        st.info("No datasets. Add one below.")

    st.text_input("New dataset name", key="new_name", placeholder="e.g. Sample B")
    st.button("➕ Add dataset", on_click=cb_add_dataset, use_container_width=True)

# ---------------- Calibration ----------------
cal, cal_err = build_calibration(log_x, log_y, [cx0, cy0, cx1, cy1])

# ---------------- Main area ----------------
img = ss.shown_img
W, H = img.size
left_col, right_col = st.columns([5, 3], gap="large")

with left_col:
    # status / instructions
    if len(ss.cal) < 3:
        st.info(f"**Calibration:** click on the image to place the "
                f"**{CAL_LABELS[len(ss.cal)]}** ({len(ss.cal)}/3 placed).")
    elif cal_err:
        st.error(cal_err)
    elif active is None:
        st.warning("Add a dataset in the sidebar before clicking data points.")
    else:
        st.success(f"Calibrated. Click to add points to **{active['name']}**.")

    # zoom controls
    z1, z2, z3 = st.columns(3)
    zoom = z1.slider("Zoom", 1.0, 12.0, 1.0, 0.5)
    pan_x = z2.slider("Pan left / right (%)", 0, 100, 50, disabled=zoom == 1.0)
    pan_y = z3.slider("Pan up / down (%)", 0, 100, 50, disabled=zoom == 1.0)

    cw, ch = W / zoom, H / zoom
    l = (W - cw) * pan_x / 100
    t = (H - ch) * pan_y / 100
    box = (int(round(l)), int(round(t)), int(round(l)) + max(1, int(round(cw))),
           int(round(t)) + max(1, int(round(ch))))
    box = (box[0], box[1], min(box[2], W), min(box[3], H))
    cw, ch = box[2] - box[0], box[3] - box[1]

    view = render_view(img, box, ss.datasets)
    click = streamlit_image_coordinates(view, key=f"click_{ss.img_id}")

    if click is not None:
        sig = tuple(sorted(click.items()))
        if sig != ss.last_click:
            ss.last_click = sig
            px = box[0] + click["x"] / click["width"] * cw
            py = box[1] + click["y"] / click["height"] * ch
            if len(ss.cal) < 3:
                ss.cal.append((px, py))
                ss.version += 1
                st.rerun()
            elif cal is not None and active is not None:
                active["points"].append((px, py))
                ss.version += 1
                st.rerun()

with right_col:
    st.subheader("Data table")
    if cal is None:
        st.caption("Complete the calibration to see the extracted values.")
    elif not ss.datasets:
        st.caption("No datasets yet.")
    else:
        st.caption("Edit values to move the points on the image; add or delete rows as needed.")
        tabs = st.tabs([d["name"] for d in ss.datasets])
        for tab, d in zip(tabs, ss.datasets):
            with tab:
                dataset_table(d, cal, x_label, y_label)

        csv_text = build_csv(ss.datasets, cal, x_label, y_label)
        st.download_button("⬇️ Download CSV", csv_text, file_name="digitized_data.csv",
                           mime="text/csv", use_container_width=True)
        with st.expander("CSV preview"):
            st.code(csv_text, language="text")
