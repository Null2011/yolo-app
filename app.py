"""
YOLOv8 画像認識アプリ (Streamlit)

起動方法:
    streamlit run app.py
"""

import io
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO

# ---------------------------------------------------------------- ページ設定
st.set_page_config(page_title="YOLO 画像認識アプリ", layout="wide")

st.title("📷 YOLO 画像認識アプリ")
st.write("画像をアップロードすると、YOLOv8 が物体を検出して表示します。")

# ---------------------------------------------------------------- サイドバー
st.sidebar.header("⚙️ 設定")

MODEL_CHOICES = {
    "yolov8n (軽量・高速)": "yolov8n.pt",
    "yolov8s (バランス)": "yolov8s.pt",
    "yolov8m (高精度)": "yolov8m.pt",
    "yolov8l (さらに高精度)": "yolov8l.pt",
    "yolov8x (最高精度・低速)": "yolov8x.pt",
}

source_mode = st.sidebar.radio(
    "重みの指定方法",
    ["プリセット（無ければ自動DL）", "ローカルの .pt ファイルを指定"],
    index=0,
)

if source_mode == "プリセット（無ければ自動DL）":
    model_label = st.sidebar.selectbox("モデル", list(MODEL_CHOICES.keys()), index=0)
    model_name = MODEL_CHOICES[model_label]
    if Path(model_name).exists():
        st.sidebar.success(f"✅ ローカルに {model_name} を確認")
    else:
        st.sidebar.info(
            f"ℹ️ {model_name} は未取得です。初回実行時に "
            f"{Path.cwd()} へ自動ダウンロードされます（要ネット接続）。"
        )
else:
    model_name = st.sidebar.text_input(
        "重みファイルのパス", value="yolov8n.pt",
        help="例: C:\\Users\\you\\models\\best.pt / 自分で学習した重みもここで指定できます",
    ).strip().strip('"')
    if not model_name:
        st.sidebar.error("パスを入力してください。")
        st.stop()
    if not Path(model_name).exists():
        st.sidebar.error(f"❌ ファイルが見つかりません: {Path(model_name).resolve()}")
        st.stop()
    st.sidebar.success(f"✅ {Path(model_name).name} を読み込みます")

confidence = st.sidebar.slider("検出の閾値 (Confidence)", 0.0, 1.0, 0.25, 0.05)
iou = st.sidebar.slider("NMS の IoU 閾値", 0.0, 1.0, 0.45, 0.05)
imgsz = st.sidebar.select_slider("推論解像度 (imgsz)", options=[320, 416, 512, 640, 960, 1280], value=640)

# ---------------------------------------------------------------- 画像処理設定
st.sidebar.markdown("---")
st.sidebar.header("🎨 画像処理")
st.sidebar.caption("補正をかけた画像に対して検出を行います。")

with st.sidebar.expander("明暗", expanded=False):
    brightness = st.slider("明るさ", 0.0, 2.0, 1.0, 0.05, help="1.0で無補正")
    contrast = st.slider("コントラスト", 0.0, 2.0, 1.0, 0.05, help="1.0で無補正")

with st.sidebar.expander("色補正", expanded=False):
    saturation = st.slider("彩度", 0.0, 2.0, 1.0, 0.05, help="0でモノクロ、1.0で無補正")
    warmth = st.slider(
        "色温度（寒色 ⇄ 暖色）", -100, 100, 0, 5,
        help="マイナスで青寄り、プラスで赤寄り",
    )

with st.sidebar.expander("モザイク", expanded=False):
    mosaic = st.slider(
        "モザイク強度", 0, 50, 0, 1,
        help="0でオフ。大きいほど粗くなります（画像全体に適用）",
    )

with st.sidebar.expander("ぼかし・シャープ", expanded=False):
    blur = st.slider(
        "ぼかし", 0.0, 10.0, 0.0, 0.5,
        help="0でオフ。ガウスぼかしの半径",
    )
    sharpen = st.slider(
        "シャープ", 0.0, 5.0, 0.0, 0.5,
        help="0でオフ。輪郭を強調します",
    )

with st.sidebar.expander("検出物体モザイク", expanded=False):
    object_mosaic = st.checkbox(
        "検出した物体にモザイクをかける", value=False,
        help="検出結果のバウンディングボックス領域だけをモザイク化します（プライバシー保護用途など）",
    )
    object_mosaic_strength = st.slider(
        "物体モザイク強度", 1, 50, 15, 1,
        help="大きいほど粗くなります",
    )

compare_mode = st.sidebar.checkbox(
    "比較モード（補正なし vs 補正あり）", value=False,
    help="元画像と補正画像の両方で検出し、結果を並べて比較します",
)
apply_to_detection = st.sidebar.checkbox(
    "補正後の画像で検出する", value=True,
    help="比較モードがオフのときのみ有効。オフにすると表示は補正・検出は元画像で行います",
    disabled=compare_mode,
)


def process_image(img: Image.Image) -> Image.Image:
    """明暗・色補正・モザイク・ぼかし・シャープを順に適用して返す。"""
    out = img

    # 明るさ・コントラスト
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)

    # 彩度
    if saturation != 1.0:
        out = ImageEnhance.Color(out).enhance(saturation)

    # 色温度（暖色/寒色）: R と B チャンネルを増減
    if warmth != 0:
        arr = np.asarray(out).astype(np.int16)
        arr[..., 0] = np.clip(arr[..., 0] + warmth, 0, 255)  # R
        arr[..., 2] = np.clip(arr[..., 2] - warmth, 0, 255)  # B
        out = Image.fromarray(arr.astype(np.uint8))

    # モザイク（縮小→拡大で画素を粗くする）
    if mosaic > 0:
        w, h = out.size
        factor = mosaic + 1
        small = out.resize(
            (max(1, w // factor), max(1, h // factor)), Image.NEAREST
        )
        out = small.resize((w, h), Image.NEAREST)

    # ぼかし（ガウス）
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur))

    # シャープ（強度を percent で反映）
    if sharpen > 0:
        out = out.filter(
            ImageFilter.UnsharpMask(radius=2, percent=int(sharpen * 100), threshold=2)
        )

    return out


# ---------------------------------------------------------------- モデル読み込み
@st.cache_resource(show_spinner="モデルを読み込み中...")
def load_model(name: str) -> YOLO:
    """初回のみ重みが自動ダウンロードされ、以後はキャッシュされる。"""
    return YOLO(name)


try:
    model = load_model(model_name)
except Exception as e:  # noqa: BLE001
    st.error(
        f"モデルの読み込みに失敗しました: {e}\n\n"
        "考えられる原因:\n"
        "1. 重みファイルが存在せず、ネット接続もない → "
        "https://github.com/ultralytics/assets/releases から .pt を手動で入手し、"
        "「ローカルの .pt ファイルを指定」で読み込んでください\n"
        "2. カレントディレクトリに書き込み権限がない\n"
        "3. 指定したファイルが YOLO の重みではない"
    )
    st.stop()

# 検出対象クラスの絞り込み（任意）
all_names = list(model.names.values())
selected_names = st.sidebar.multiselect(
    "検出するクラス（未選択なら全クラス）", all_names, default=[]
)
class_ids = [i for i, n in model.names.items() if n in selected_names] or None


def run_detection(src_image):
    """1枚に対して検出を実行し、結果オブジェクトを返す。"""
    res = model.predict(
        source=src_image,
        conf=confidence,
        iou=iou,
        imgsz=imgsz,
        classes=class_ids,
        verbose=False,
    )
    return res[0]


def result_to_df(result):
    """検出結果を DataFrame に変換する。"""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame()
    rows = []
    for i, box in enumerate(boxes, start=1):
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
        rows.append(
            {
                "No.": i,
                "クラス": model.names[class_id],
                "信頼度": round(float(box.conf[0]), 3),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "幅": x2 - x1,
                "高さ": y2 - y1,
            }
        )
    return pd.DataFrame(rows)


def plotted_rgb(result):
    """検出を描画した RGB 画像（NumPy 配列）を返す。"""
    return result.plot()[:, :, ::-1]


def mosaic_objects(src_image, result, strength):
    """検出ボックス領域だけをモザイク化した RGB 画像（NumPy 配列）を返す。

    src_image: 検出をかけた元の PIL 画像（描画なし）
    result:    その検出結果
    strength:  モザイクの粗さ（大きいほど粗い）
    """
    arr = np.asarray(src_image.convert("RGB")).copy()
    h, w = arr.shape[:2]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return arr

    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
        # 画像範囲内にクリップ
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        region = Image.fromarray(arr[y1:y2, x1:x2])
        rw, rh = region.size
        factor = strength + 1
        small = region.resize(
            (max(1, rw // factor), max(1, rh // factor)), Image.NEAREST
        )
        arr[y1:y2, x1:x2] = np.asarray(small.resize((rw, rh), Image.NEAREST))

    return arr

# ---------------------------------------------------------------- 画像入力
uploaded_file = st.file_uploader(
    "画像を選択してください...", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is None:
    st.info("👆 画像をアップロードすると推論が始まります。")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
processed = process_image(image)
stem = uploaded_file.name.rsplit(".", 1)[0]


def render_detection_list(df, result, key_prefix):
    """検出一覧タブ（詳細・集計）を描画する。"""
    if df.empty:
        st.warning("オブジェクトが検出されませんでした。閾値を下げてみてください。")
        return
    tab1, tab2 = st.tabs(["📋 詳細", "📊 クラス別集計"])
    with tab1:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "CSV でダウンロード",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"detections_{key_prefix}.csv",
            mime="text/csv",
            key=f"csv_{key_prefix}",
        )
    with tab2:
        counts = Counter(df["クラス"])
        summary = (
            pd.DataFrame({"クラス": list(counts.keys()), "個数": list(counts.values())})
            .sort_values("個数", ascending=False)
            .reset_index(drop=True)
        )
        summary["平均信頼度"] = summary["クラス"].map(
            df.groupby("クラス")["信頼度"].mean().round(3)
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.bar_chart(summary.set_index("クラス")["個数"])
    st.caption(f"検出数: {len(df)} 件 / 推論時間: {result.speed['inference']:.1f} ms")


if compare_mode:
    # ========================= 比較モード =========================
    with st.spinner("推論中（2枚）..."):
        result_orig = run_detection(image)
        result_proc = run_detection(processed)

    df_orig = result_to_df(result_orig)
    df_proc = result_to_df(result_proc)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("補正なし")
        st.image(plotted_rgb(result_orig), use_container_width=True)
    with col2:
        st.subheader("補正あり")
        st.image(plotted_rgb(result_proc), use_container_width=True)

    # 検出物体モザイク（補正なし・補正ありの両方）
    if object_mosaic:
        st.markdown("---")
        st.subheader("🟦 検出物体モザイク")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**補正なし**")
            m_orig = mosaic_objects(image, result_orig, object_mosaic_strength)
            st.image(m_orig, use_container_width=True)
            b1 = io.BytesIO()
            Image.fromarray(m_orig).save(b1, format="PNG")
            st.download_button(
                "保存 (PNG)", data=b1.getvalue(),
                file_name=f"mosaic_orig_{stem}.png", mime="image/png",
                key="mosaic_dl_orig",
            )
        with mc2:
            st.markdown("**補正あり**")
            m_proc = mosaic_objects(processed, result_proc, object_mosaic_strength)
            st.image(m_proc, use_container_width=True)
            b2 = io.BytesIO()
            Image.fromarray(m_proc).save(b2, format="PNG")
            st.download_button(
                "保存 (PNG)", data=b2.getvalue(),
                file_name=f"mosaic_proc_{stem}.png", mime="image/png",
                key="mosaic_dl_proc",
            )

    # 検出数の差をひと目で
    st.markdown("---")
    st.subheader("📊 検出数の比較")
    m1, m2, m3 = st.columns(3)
    m1.metric("補正なし 検出数", len(df_orig))
    m2.metric("補正あり 検出数", len(df_proc))
    m3.metric("差", len(df_proc) - len(df_orig), delta=len(df_proc) - len(df_orig))

    # クラス別の増減
    c_orig = Counter(df_orig["クラス"]) if not df_orig.empty else Counter()
    c_proc = Counter(df_proc["クラス"]) if not df_proc.empty else Counter()
    all_cls = sorted(set(c_orig) | set(c_proc))
    if all_cls:
        cmp_df = pd.DataFrame(
            {
                "クラス": all_cls,
                "補正なし": [c_orig.get(c, 0) for c in all_cls],
                "補正あり": [c_proc.get(c, 0) for c in all_cls],
            }
        )
        cmp_df["増減"] = cmp_df["補正あり"] - cmp_df["補正なし"]
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        st.bar_chart(cmp_df.set_index("クラス")[["補正なし", "補正あり"]])

    # それぞれの詳細一覧
    st.markdown("---")
    st.subheader("🔍 検出一覧")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**補正なし**")
        render_detection_list(df_orig, result_orig, "orig")
    with d2:
        st.markdown("**補正あり**")
        render_detection_list(df_proc, result_proc, "proc")

else:
    # ========================= 通常モード =========================
    detect_source = processed if apply_to_detection else image

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("補正後の画像")
        st.image(processed, use_container_width=True)
        buf_proc = io.BytesIO()
        processed.save(buf_proc, format="PNG")
        st.download_button(
            "補正画像を保存 (PNG)",
            data=buf_proc.getvalue(),
            file_name=f"edited_{stem}.png",
            mime="image/png",
        )

    with col2:
        st.subheader("検出結果")
        with st.spinner("推論中..."):
            result = run_detection(detect_source)
        res_plotted = plotted_rgb(result)
        st.image(res_plotted, use_container_width=True)
        buf = io.BytesIO()
        Image.fromarray(res_plotted).save(buf, format="PNG")
        st.download_button(
            "検出結果を保存 (PNG)",
            data=buf.getvalue(),
            file_name=f"detected_{stem}.png",
            mime="image/png",
        )

    # 検出物体モザイク
    if object_mosaic:
        st.markdown("---")
        st.subheader("🟦 検出物体モザイク")
        mosaic_arr = mosaic_objects(detect_source, result, object_mosaic_strength)
        st.image(mosaic_arr, use_container_width=True)
        buf_m = io.BytesIO()
        Image.fromarray(mosaic_arr).save(buf_m, format="PNG")
        st.download_button(
            "モザイク画像を保存 (PNG)",
            data=buf_m.getvalue(),
            file_name=f"mosaic_{stem}.png",
            mime="image/png",
        )

    st.markdown("---")
    st.subheader("🔍 検出されたオブジェクト一覧")
    render_detection_list(result_to_df(result), result, "single")
