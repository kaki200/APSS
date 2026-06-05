import cv2
import numpy as np
import os
import time
import threading
import json
import base64
import socket
import struct
import queue
import tkinter as tk
from PIL import Image, ImageTk
from ftplib import FTP
import win32clipboard
from http.server import BaseHTTPRequestHandler, HTTPServer

# 設定ファイルと画像保存ディレクトリの定義
CONFIG_FILE = "scanner_config.json"
LOCAL_DIR = "./scanned_images"
ORIG_DIR = "./original_images"

# ディレクトリが存在しない場合は作成
os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(ORIG_DIR, exist_ok=True)

# アプリケーションのデフォルト設定
config = {
    "TRANSFER_MODE": "FTP",
    "FTP_HOST": "",
    "FTP_PORT": 21,
    "FTP_USER": "",
    "FTP_PASS": "",
    "FTP_DIR": "/",
    "HTTP_PORT": 8080,
    "ALWAYS_ON_TOP": False,
    "ASPECT_RATIO": "Auto",
    "TARGET_MODE": "Auto",
    "AREA_TH": 5,
    "APPROX_TH": 0.03,
    "CANNY_MIN": 30,
    "CANNY_MAX": 150,
    "PROJ_THRESH": 120,
    "ADAPTIVE_BLOCK": 21,
    "ADAPTIVE_C": 5,
    "BLACK_LIMIT": 30,
    "GAMMA": 1.2,
    "SMOOTH": 9
}

# スレッド間通信用のキューと、Ping監視ステータス保持用の辞書
gui_queue = queue.Queue()
ping_status = {"status": "待機中...", "ping": "--- ms", "color": "orange"}
http_server_error = False

def load_config():
    """設定ファイル(JSON)が存在する場合は読み込み、なければデフォルト設定で新規作成する"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if "FTP_PASS" in loaded and loaded["FTP_PASS"]:
                loaded["FTP_PASS"] = base64.b64decode(loaded["FTP_PASS"].encode('utf-8')).decode('utf-8')
            config.update(loaded)
    else:
        save_config()

def save_config():
    """現在の設定をJSONファイルとして保存する"""
    save_data = config.copy()
    if save_data.get("FTP_PASS"):
        # パスワードをBase64でエンコードして保存（平文を見えなくする）
        save_data["FTP_PASS"] = base64.b64encode(save_data["FTP_PASS"].encode('utf-8')).decode('utf-8')
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4)

def copy_file_to_clipboard(filepath):
    """指定されたファイルをWindowsのクリップボードにコピー（ファイルドロップ形式）する"""
    time.sleep(0.3) 
    abs_path = os.path.abspath(filepath)
    wide_path = abs_path.encode('utf-16le') + b'\0\0'
    dropfiles = struct.pack('IIIIi', 20, 0, 0, 0, 1)
    data = dropfiles + wide_path
    
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_HDROP, data)
    win32clipboard.CloseClipboard()

# --- 四角形検出ロジック（ベゼル内側判定を追加） ---
def get_screen_points(img, area_percent, approx_val, canny_min, canny_max, adapt_block, adapt_c, proj_thresh, target_mode):
    """画像からスクリーン（四角形）の頂点4つを検出する"""
    # 処理速度向上のため、画像を高さ800pxにリサイズして処理を行う
    ratio = 800.0 / img.shape[0]
    dim = (int(img.shape[1] * ratio), 800)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 画面いっぱいの被写体を閉じた輪郭にするため、画像の周囲に黒いパディングを一時的に追加
    pad = 10
    padded_gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    img_area = padded_gray.shape[0] * padded_gray.shape[1]
    
    def extract_quads(binary_img):
        """二値化画像から四角形の輪郭を抽出する内部関数"""
        # 階層構造（RETR_TREE）を取得して、外枠と内枠を見分ける
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None: return [], []
        hierarchy = hierarchy[0]
        
        quads_info = []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < (img_area * (area_percent / 100.0)):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, approx_val * peri, True)
            if len(approx) == 4:
                quads_info.append({"approx": approx, "idx": i, "area": area})
                
        # ベゼル（黒枠）のような太い枠は、外側の境界線（親）と内側の境界線（子）の2つの四角形を生む。
        # 外枠を除外して内枠（スクリーン部分）だけを残すロジック
        final_quads = []
        quad_indices = set(q["idx"] for q in quads_info)
        
        for q in quads_info:
            idx = q["idx"]
            child_idx = hierarchy[idx][2] # 最初の子要素のインデックス
            
            has_quad_child = False
            curr_child = child_idx
            # 子要素やその兄弟要素に「四角形」が存在するかチェック
            while curr_child != -1:
                if curr_child in quad_indices:
                    has_quad_child = True
                    break
                curr_child = hierarchy[curr_child][0] # 次の兄弟要素
                
            # 子に四角形を持たない（＝自分が一番内側の四角形である）場合のみ採用
            if not has_quad_child:
                final_quads.append(q["approx"])
                
        return final_quads, contours

    all_quads = []
    all_contours = []

    # プロジェクター向け検出処理：Cannyエッジと明度閾値を使用
    if target_mode in ("Auto", "Projector"):
        blurred_p = cv2.GaussianBlur(padded_gray, (5, 5), 0)
        
        # 1. 既存：Cannyエッジ検出
        edged_p = cv2.Canny(blurred_p, canny_min, canny_max)
        q_p, c_p = extract_quads(edged_p)
        all_quads.extend(q_p)
        if c_p: all_contours.append(max(c_p, key=cv2.contourArea))
        
        # 2. 新規：明るさの閾値（明度）による映像領域の切り出し
        if proj_thresh > 0:
            _, thresh_p = cv2.threshold(blurred_p, proj_thresh, 255, cv2.THRESH_BINARY)
            q_p2, c_p2 = extract_quads(thresh_p)
            all_quads.extend(q_p2)
            if c_p2: all_contours.append(max(c_p2, key=cv2.contourArea))

    # ディスプレイ向け検出処理：適応的閾値処理（Adaptive Threshold）を使用
    if target_mode in ("Auto", "Display"):
        blurred_d = cv2.GaussianBlur(padded_gray, (5, 5), 0)
        if adapt_block % 2 == 0: adapt_block += 1
        if adapt_block < 3: adapt_block = 3
        
        thresh_d = cv2.adaptiveThreshold(blurred_d, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_block, adapt_c)
        q_d, c_d = extract_quads(thresh_d)
        all_quads.extend(q_d)
        if c_d: all_contours.append(max(c_d, key=cv2.contourArea))

    screen_contour = None
    
    # 四角形が見つかった場合は最適なものを採用し、見つからない場合は最大面積の輪郭から四角形を近似する
    if all_quads:
        if target_mode in ("Auto", "Projector"):
            best_quad = None
            best_score = -1
            
            h, w = padded_gray.shape[:2]
            center = np.array([w / 2, h / 2])
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            
            for q in all_quads:
                mask = np.zeros(padded_gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [q], 255)
                
                # 四角形の境界（枠線）を太さ30pxで描画（内側に15px、外側に15px）
                mask_edge = np.zeros(padded_gray.shape, dtype=np.uint8)
                cv2.polylines(mask_edge, [q], isClosed=True, color=255, thickness=30)
                
                # 枠の「すぐ内側」の明るさ（映像なら明るく、余白なら暗い）
                mask_inner_edge = cv2.bitwise_and(mask_edge, mask)
                mean_val_inner_edge = cv2.mean(padded_gray, mask=mask_inner_edge)[0]
                
                # 枠の「すぐ外側」の明るさ
                mask_outer_edge = cv2.bitwise_and(mask_edge, cv2.bitwise_not(mask))
                mean_val_outer_edge = cv2.mean(padded_gray, mask=mask_outer_edge)[0]
                
                # 投影されていない暗い余白を避けるため、「すぐ内側」が明るく「すぐ外側」が暗いことを強く評価
                contrast_factor = max(0.1, mean_val_inner_edge - mean_val_outer_edge)
                
                # 画像中心からどれくらい離れているかをペナルティ化
                M = cv2.moments(q)
                cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else w // 2
                cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else h // 2
                dist_factor = max(0.1, 1.0 - (np.linalg.norm(np.array([cx, cy]) - center) / max_dist))
                
                # 大きな物理枠が勝たないように面積の影響を下げ、エッジの明暗差の重みを大幅に上げる
                score = cv2.contourArea(q) * (mean_val_inner_edge ** 2) * dist_factor * (contrast_factor ** 3)
                if score > best_score:
                    best_score = score
                    best_quad = q
            screen_contour = best_quad
        else:
            screen_contour = max(all_quads, key=cv2.contourArea)
    elif all_contours:
        if target_mode in ("Auto", "Projector"):
            best_c = None
            best_score = -1
            
            h, w = padded_gray.shape[:2]
            center = np.array([w / 2, h / 2])
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            
            for c in all_contours:
                mask = np.zeros(padded_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
                
                mask_edge = np.zeros(padded_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask_edge, [c], -1, 255, thickness=30)
                
                mask_inner_edge = cv2.bitwise_and(mask_edge, mask)
                mean_val_inner_edge = cv2.mean(padded_gray, mask=mask_inner_edge)[0]
                
                mask_outer_edge = cv2.bitwise_and(mask_edge, cv2.bitwise_not(mask))
                mean_val_outer_edge = cv2.mean(padded_gray, mask=mask_outer_edge)[0]
                
                contrast_factor = max(0.1, mean_val_inner_edge - mean_val_outer_edge)
                
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else w // 2
                cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else h // 2
                dist_factor = max(0.1, 1.0 - (np.linalg.norm(np.array([cx, cy]) - center) / max_dist))
                
                score = cv2.contourArea(c) * (mean_val_inner_edge ** 2) * dist_factor * (contrast_factor ** 3)
                if score > best_score:
                    best_score = score
                    best_c = c
        else:
            best_c = max(all_contours, key=cv2.contourArea)
            
        if best_c is not None and cv2.contourArea(best_c) > (img_area * 0.05):
            rect = cv2.minAreaRect(best_c)
            box = cv2.boxPoints(rect)
            screen_contour = np.int32(box).reshape(4, 1, 2)

    if screen_contour is not None:
        # パディングとして追加した座標分を元に戻す
        screen_contour = screen_contour - pad
        # 座標が画像サイズをはみ出さないようにクリップする
        h, w = resized.shape[:2]
        screen_contour[:, 0, 0] = np.clip(screen_contour[:, 0, 0], 0, w)
        screen_contour[:, 0, 1] = np.clip(screen_contour[:, 0, 1], 0, h)
    else:
        # それでも見つからない場合は、画像より少し小さいデフォルトの矩形を返す
        h, w = resized.shape[:2]
        screen_contour = np.array([[[50, 50]], [[w-50, 50]], [[w-50, h-50]], [[50, h-50]]])

    # リサイズ前の元の座標スケールに戻して返す
    return screen_contour.reshape(4, 2) / ratio

def apply_correction_and_save(orig, pts_original_scale, output_path):
    """台形補正（射影変換）と画質調整（色調・ガンマ補正等）を行い、画像を保存する"""
    def order_points(pts):
        """4つの頂点を 左上、右上、右下、左下 の順序に並び替える"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(pts_original_scale)
    (tl, tr, br, bl) = rect

    # 変形後の画像サイズ（最大幅・最大高さ）を計算
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    info_msg = ""
    aspect_mode = config.get("ASPECT_RATIO", "Auto")
    if aspect_mode == "Auto" and maxHeight > 0:
        # 消失点とカメラの焦点距離を用いた本来のアスペクト比の数学的推定
        def estimate_aspect_ratio():
            h, w = orig.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            
            def line_intersection(p1, p2, p3, p4):
                denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
                if denom == 0: return None
                px = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[0] - p4[0]) - (p1[0] - p2[0])*(p3[0]*p4[1] - p3[1]*p4[0])) / denom
                py = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0]*p4[1] - p3[1]*p4[0])) / denom
                return np.array([px, py])

            vp_h = line_intersection(tl, tr, bl, br)
            vp_v = line_intersection(tl, bl, tr, br)
            if vp_h is None or vp_v is None: return None

            v1 = vp_h - [cx, cy]
            v2 = vp_v - [cx, cy]
            f2 = -np.dot(v1, v2)
            # レンズの歪みや、元の図形が長方形でない等で推定不能な場合はスキップ
            if f2 <= 0: return None
            f = np.sqrt(f2)

            V_h = np.array([v1[0], v1[1], f])
            V_v = np.array([v2[0], v2[1], f])
            N = np.cross(V_h / np.linalg.norm(V_h), V_v / np.linalg.norm(V_v))
            if np.linalg.norm(N) == 0: return None
            N = N / np.linalg.norm(N)

            def intersect_plane(pt2d):
                ray = np.array([pt2d[0] - cx, pt2d[1] - cy, f])
                dot_val = np.dot(ray, N)
                return ray * (1.0 / dot_val) if dot_val != 0 else ray

            TL = intersect_plane(tl)
            TR = intersect_plane(tr)
            BL = intersect_plane(bl)
            
            return np.linalg.norm(TR - TL) / np.linalg.norm(BL - TL) if np.linalg.norm(BL - TL) > 0 else None

        estimated_ratio = estimate_aspect_ratio()
        if estimated_ratio is not None:
            # 面積（ピクセル数）を維持したまま、推定された正しい縦横比を適用する
            area = maxWidth * maxHeight
            maxHeight = int(np.sqrt(area / estimated_ratio))
            maxWidth = int(maxHeight * estimated_ratio)
            
            info_msg = f" [推定比 {estimated_ratio * 9:.2f}:9]"
            
    elif aspect_mode != "Auto" and maxHeight > 0:
        # ユーザー指定の縦横比（アスペクト比）に合わせて出力サイズを強制する
        is_landscape = maxWidth > maxHeight
        if aspect_mode == "16:9":
            ratio = 16.0 / 9.0
        elif aspect_mode == "4:3":
            ratio = 4.0 / 3.0
        elif aspect_mode == "A4":
            ratio = 1.414 # 白銀比 (1:√2)
        else:
            ratio = maxWidth / maxHeight

        if is_landscape:
            maxHeight = int(maxWidth / ratio)
        else:
            maxWidth = int(maxHeight / ratio)

    # 射影変換（パースペクティブ変換）を実行
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # ホワイトバランスの自動調整（上位0.5%の明るさのピクセルを白とみなしてゲインを計算）
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    p_high = np.percentile(gray_w, 99.5)
    white_pixels = warped[gray_w >= p_high]
    b_mean, g_mean, r_mean = np.mean(white_pixels, axis=0)
    
    b_gain = 255.0 / b_mean if b_mean > 0 else 1.0
    g_gain = 255.0 / g_mean if g_mean > 0 else 1.0
    r_gain = 255.0 / r_mean if r_mean > 0 else 1.0
    
    b, g, r = cv2.split(warped)
    b = np.clip(b * b_gain, 0, 255).astype(np.uint8)
    g = np.clip(g * g_gain, 0, 255).astype(np.uint8)
    r = np.clip(r * r_gain, 0, 255).astype(np.uint8)
    wb_adjusted = cv2.merge((b, g, r))

    # 黒レベルの引き締め（指定した割合以下の暗い部分を真っ黒にする）
    gray_wb = cv2.cvtColor(wb_adjusted, cv2.COLOR_BGR2GRAY)
    dynamic_black = np.percentile(gray_wb, 1)
    
    limit = config["BLACK_LIMIT"]
    if dynamic_black > limit: dynamic_black = limit 

    # LUT（ルックアップテーブル）を利用したガンマ補正とコントラスト調整
    lut = np.zeros((256,), dtype=np.uint8)
    gamma = config["GAMMA"]
    for i in range(256):
        if i < dynamic_black: val = 0
        elif i > 255: val = 255
        else: val = np.clip(255.0 * (i - dynamic_black) / (255 - dynamic_black), 0, 255)
        lut[i] = np.clip(pow(val / 255.0, gamma) * 255.0, 0, 255)
        
    adjusted = cv2.LUT(wb_adjusted, lut)

    # ノイズ平滑化（バイラテラルフィルタ：エッジを残しつつノイズをぼかす）
    smooth = config["SMOOTH"]
    if smooth > 0:
        adjusted = cv2.bilateralFilter(adjusted, d=smooth, sigmaColor=75, sigmaSpace=75)

    # クリップボード用のテンポラリファイルとして保存
    temp_path = os.path.join(LOCAL_DIR, "temp.jpg")
    cv2.imwrite(temp_path, adjusted)
    
    # 常に履歴として保存
    cv2.imwrite(output_path, adjusted)
    return info_msg

class MacroDroidHTTPHandler(BaseHTTPRequestHandler):
    """MacroDroidからのHTTP POSTリクエスト（画像データ）を受け取るハンドラ"""

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            content_type = self.headers.get('Content-Type', '')
            post_data = self.rfile.read(content_length)

            file_data = None
            # MacroDroidからのファイル送信(multipart/form-data)を解析
            if 'multipart/form-data' in content_type:
                try:
                    boundary = content_type.split("boundary=")[1].split(";")[0].strip(' "').encode()
                    parts = post_data.split(boundary)
                    for part in parts:
                        if b'filename=' in part:
                            if b'\r\n\r\n' in part:
                                file_data = part.split(b'\r\n\r\n', 1)[1]
                                if file_data.endswith(b'\r\n--\r\n'): file_data = file_data[:-6]
                                elif file_data.endswith(b'\r\n--'): file_data = file_data[:-4]
                                elif file_data.endswith(b'\r\n'): file_data = file_data[:-2]
                                break
                except Exception:
                    pass
            else:
                file_data = post_data

            if file_data and len(file_data) > 100:
                file_name = f"macrodroid_{int(time.time())}.jpg"
                orig_path = os.path.join(ORIG_DIR, file_name)
                with open(orig_path, 'wb') as f:
                    f.write(file_data)
                
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
                
                gui_queue.put({"type": "status", "message": f"⬇ HTTP受信: {file_name}", "color": "blue"})
                
                img = cv2.imread(orig_path)
                if img is not None:
                    start_time = time.time()
                    pts_orig = get_screen_points(img, config["AREA_TH"], config["APPROX_TH"], config["CANNY_MIN"], config["CANNY_MAX"], config["ADAPTIVE_BLOCK"], config["ADAPTIVE_C"], config.get("PROJ_THRESH", 120), config["TARGET_MODE"])
                    output_path = os.path.join(LOCAL_DIR, os.path.splitext(file_name)[0] + "_processed.jpg")
                    gui_queue.put({
                        "type": "new",
                        "orig_path": orig_path,
                        "output_path": output_path,
                        "pts_orig": pts_orig,
                        "start_time": start_time
                    })
                else:
                    # 不完全な一時ファイルが送られた場合はエラーにせず静かに無視する
                    pass
            else:
                self.send_response(200) # MacroDroidにエラー通知を出させないため常に200 OKを返す
                self.end_headers()
                self.wfile.write(b"Ignored empty file")
                if file_data:
                    preview = file_data[:30].decode('utf-8', 'ignore').replace('\r', '').replace('\n', '')
                    gui_queue.put({"type": "status", "message": f"❌ 受信エラー: 文字が届いています ({preview}…)", "color": "red"})
                else:
                    gui_queue.put({"type": "status", "message": "❌ 受信エラー: データ空", "color": "red"})

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())
            gui_queue.put({"type": "status", "message": "❌ HTTP処理エラー", "color": "red"})

    def log_message(self, format, *args):
        pass # コンソールのログ出力を抑制

def http_monitor_thread():
    """MacroDroid用のHTTPサーバーを起動する別スレッド"""
    global http_server_error
    server = None
    current_port = -1
    while True:
        if config.get("TRANSFER_MODE", "FTP") == "HTTP":
            port = config.get("HTTP_PORT", 8080)
            if server is None or current_port != port:
                if server: server.server_close()
                try:
                    server = HTTPServer(('0.0.0.0', port), MacroDroidHTTPHandler)
                    server.timeout = 2
                    current_port = port
                    http_server_error = False
                except Exception:
                    server = None
                    http_server_error = True
                    time.sleep(2)
            if server:
                server.handle_request() # タイムアウト(2秒)付きでリクエストを待つ
        else:
            if server:
                server.server_close()
                server = None
            http_server_error = False
            time.sleep(2)

def ping_monitor_thread():
    """ネットワークの監視とIP/ポート表示を行う別スレッド"""
    while True:
        if config.get("TRANSFER_MODE", "FTP") == "FTP":
            start_time = time.time()
            try:
                s = socket.create_connection((config["FTP_HOST"], config["FTP_PORT"]), 2)
                s.close()
                ping_ms = int((time.time() - start_time) * 1000)
                ping_status["ping"] = f"{ping_ms} ms"
                ping_status["status"] = "✅ FTP接続OK (監視中)"
                ping_status["color"] = "green"
            except Exception:
                ping_status["ping"] = "エラー"
                ping_status["status"] = "❌ FTP接続失敗・検索中..."
                ping_status["color"] = "red"
        else:
            # 自身のIPアドレスを取得して表示
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip_addr = s.getsockname()[0]
                s.close()
            except Exception:
                ip_addr = "不明"
            port = config.get("HTTP_PORT", 8080)
            
            if http_server_error:
                ping_status["ping"] = f"Port: {port}"
                ping_status["status"] = "❌ HTTP起動エラー・ポート競合"
                ping_status["color"] = "red"
            else:
                ping_status["ping"] = f"{ip_addr}:{port}"
                ping_status["status"] = "✅ HTTP待機中 (MacroDroid)"
                ping_status["color"] = "green"
            
        time.sleep(2)

def ftp_monitor_thread():
    """FTPサーバーを監視し、新規画像があればダウンロードして自動処理を行う別スレッド"""
    seen_files = set()
    is_first_check = True
    
    while True:
        if config.get("TRANSFER_MODE", "FTP") != "FTP":
            time.sleep(2)
            continue
            
        try:
            ftp = FTP()
            ftp.connect(config["FTP_HOST"], config["FTP_PORT"], timeout=15)
            ftp.login(config["FTP_USER"], config["FTP_PASS"])
            ftp.cwd(config["FTP_DIR"])
            
            files = [f for f in ftp.nlst() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 初回起動時は既存ファイルを無視（処理済みとして記録）する
            if is_first_check:
                seen_files = set(files)
                is_first_check = False
            else:
                for file_name in files:
                    if file_name not in seen_files:
                        seen_files.add(file_name)
                        
                        gui_queue.put({"type": "status", "message": f"⬇ 受信中: {file_name}", "color": "blue"})

                        orig_path = os.path.join(ORIG_DIR, file_name)
                        with open(orig_path, 'wb') as f:
                            ftp.retrbinary(f'RETR {file_name}', f.write)
                        
                        gui_queue.put({"type": "status", "message": "⚙ 画像処理中...", "color": "orange"})

                        img = cv2.imread(orig_path)
                        # 画像処理と台形補正の適用
                        if img is not None:
                            start_time = time.time()
                            
                            pts_orig = get_screen_points(img, config["AREA_TH"], config["APPROX_TH"], config["CANNY_MIN"], config["CANNY_MAX"], config["ADAPTIVE_BLOCK"], config["ADAPTIVE_C"], config.get("PROJ_THRESH", 120), config["TARGET_MODE"])
                            output_path = os.path.join(LOCAL_DIR, os.path.splitext(file_name)[0] + "_processed.jpg")

                            # GUIへ枠検出結果を最優先で送る (重い補正・保存はGUI描画後に裏で行う)
                            gui_queue.put({
                                "type": "new",
                                "orig_path": orig_path,
                                "output_path": output_path,
                                "pts_orig": pts_orig,
                                "start_time": start_time
                            })
            ftp.quit()
        except Exception:
            pass
        time.sleep(1)

class ScannerApp:
    """GUIアプリケーション（Tkinter）のメインクラス"""
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Scanner Dashboard")
        self.root.geometry("1100x950") 
        
        self.current_orig_img = None
        self.current_output_path = None
        self.canvas_scale = 1.0
        
        self.drag_mode = None 
        self.drag_idx = -1
        self.last_x = 0
        self.last_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_start_pts = None
        self.current_snap_points = []
        
        self.base_disp_img = None
        self._preview_timer = None
        self._detect_timer = None

        # 左右のフレーム分割（左:設定パネル, 右:画像プレビュー＆サムネイル）
        self.left_frame = tk.Frame(root, width=380, padx=10, pady=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.right_frame = tk.Frame(root, bg="gray")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_settings_ui()
        self.setup_canvas_ui()
        self.setup_thumbnail_ui()
        self.apply_topmost()

        threading.Thread(target=ping_monitor_thread, daemon=True).start()
        threading.Thread(target=ftp_monitor_thread, daemon=True).start()
        threading.Thread(target=http_monitor_thread, daemon=True).start()

        self.update_gui_loop()

    def setup_settings_ui(self):
        """左側の設定画面（各種パラメータのスライダーや入力欄）のUI構築"""
        self.setting_canvas = tk.Canvas(self.left_frame, highlightthickness=0)
        self.setting_scrollbar = tk.Scrollbar(self.left_frame, orient=tk.VERTICAL, command=self.setting_canvas.yview)
        self.setting_inner = tk.Frame(self.setting_canvas)
        
        self.setting_inner.bind("<Configure>", lambda e: self.setting_canvas.configure(scrollregion=self.setting_canvas.bbox("all")))
        self.setting_canvas.create_window((0, 0), window=self.setting_inner, anchor="nw")
        self.setting_canvas.configure(yscrollcommand=self.setting_scrollbar.set)
        
        self.setting_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.setting_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.setting_canvas.bind_all("<MouseWheel>", lambda e: self.setting_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        target = self.setting_inner

        tk.Label(target, text="【ステータス】", font=("", 11, "bold")).pack(pady=(0, 2))
        self.status_lbl = tk.Label(target, text="起動中...", font=("", 10, "bold"))
        self.status_lbl.pack()
        self.ping_lbl = tk.Label(target, text="Ping: ---")
        self.ping_lbl.pack(pady=(0, 5))
        
        self.process_status_lbl = tk.Label(target, text="待機中", font=("", 10, "bold"), fg="gray")
        self.process_status_lbl.pack(pady=(0, 10))

        tk.Label(target, text="【通信設定】", font=("", 11, "bold")).pack(pady=(10, 2))
        
        mode_frame = tk.Frame(target)
        mode_frame.pack(fill=tk.X, pady=2)
        self.transfer_mode_var = tk.StringVar(value=config.get("TRANSFER_MODE", "FTP"))
        tk.Radiobutton(mode_frame, text="FTP監視", variable=self.transfer_mode_var, value="FTP", command=self.on_transfer_mode_change).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="HTTP受信(MacroDroid)", variable=self.transfer_mode_var, value="HTTP", command=self.on_transfer_mode_change).pack(side=tk.LEFT)

        self.conn_container = tk.Frame(target)
        self.conn_container.pack(fill=tk.X)
        
        self.ftp_frame = tk.Frame(self.conn_container)
        self.http_frame = tk.Frame(self.conn_container)
        
        self.entries = {}
        
        ftp_fields = [
            ("FTP_HOST", "ホスト (IP):"), ("FTP_PORT", "ポート:"),
            ("FTP_USER", "ユーザー名:"), ("FTP_PASS", "パスワード:"),
            ("FTP_DIR", "監視フォルダ:")
        ]
        for key, label in ftp_fields:
            frame = tk.Frame(self.ftp_frame)
            frame.pack(fill=tk.X, pady=1)
            tk.Label(frame, text=label, width=12, anchor="e").pack(side=tk.LEFT)
            ent = tk.Entry(frame)
            ent.insert(0, str(config.get(key, "")))
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if key == "FTP_PASS":
                ent.config(show="*")
                ent.bind("<FocusIn>", lambda e, w=ent: w.config(show=""))
                ent.bind("<FocusOut>", lambda e, w=ent: w.config(show="*"))
            self.entries[key] = ent
            
        frame = tk.Frame(self.http_frame)
        frame.pack(fill=tk.X, pady=1)
        tk.Label(frame, text="待受ポート:", width=12, anchor="e").pack(side=tk.LEFT)
        ent = tk.Entry(frame)
        ent.insert(0, str(config.get("HTTP_PORT", 8080)))
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entries["HTTP_PORT"] = ent
            
        tk.Button(target, text="設定保存", command=self.save_settings, bg="lightblue").pack(fill=tk.X, pady=(5, 10))
        self.on_transfer_mode_change()

        tk.Label(target, text="【動作オプション】", font=("", 11, "bold")).pack(pady=2)
        self.topmost_var = tk.BooleanVar(value=config.get("ALWAYS_ON_TOP", False))
        tk.Checkbutton(target, text="ウィンドウを常に最前面", variable=self.topmost_var, command=self.apply_topmost).pack(anchor=tk.W)
        # 手動切り抜き・履歴保存のチェックボックスをUIから削除（内部で常にONとして処理）

        tk.Label(target, text="【出力設定】", font=("", 11, "bold")).pack(pady=(10, 2))
        
        aspect_frame = tk.Frame(target)
        aspect_frame.pack(fill=tk.X, pady=2)
        tk.Label(aspect_frame, text="縦横比: ").pack(side=tk.LEFT)
        self.aspect_var = tk.StringVar(value=config.get("ASPECT_RATIO", "Auto"))
        for val in ["Auto", "16:9", "4:3", "A4"]:
            tk.Radiobutton(aspect_frame, text="自動" if val=="Auto" else val, variable=self.aspect_var, value=val, command=self.on_aspect_change).pack(side=tk.LEFT)

        tk.Label(target, text="【枠の検出設定】", font=("", 11, "bold")).pack(pady=(10, 2))
        
        mode_frame = tk.Frame(target)
        mode_frame.pack(fill=tk.X, pady=2)
        tk.Label(mode_frame, text="対象: ").pack(side=tk.LEFT)
        self.target_mode_var = tk.StringVar(value=config["TARGET_MODE"])
        tk.Radiobutton(mode_frame, text="Auto", variable=self.target_mode_var, value="Auto", command=self.on_mode_change).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="プロジェクター", variable=self.target_mode_var, value="Projector", command=self.on_mode_change).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="ディスプレイ", variable=self.target_mode_var, value="Display", command=self.on_mode_change).pack(side=tk.LEFT)

        self.dynamic_detect_container = tk.Frame(target)
        self.dynamic_detect_container.pack(fill=tk.X)

        self.detect_common_frame = tk.Frame(self.dynamic_detect_container)
        self.detect_common_frame.pack(fill=tk.X)
        self.area_var = tk.IntVar(value=config["AREA_TH"])
        tk.Scale(self.detect_common_frame, variable=self.area_var, from_=1, to=99, orient=tk.HORIZONTAL, label="面積の足切り % (小図形無視)", command=self.on_detect_slider_move).pack(fill=tk.X)
        self.approx_var = tk.DoubleVar(value=config["APPROX_TH"])
        tk.Scale(self.detect_common_frame, variable=self.approx_var, from_=0.01, to=0.10, resolution=0.01, orient=tk.HORIZONTAL, label="ゆがみ許容度", command=self.on_detect_slider_move).pack(fill=tk.X)
        
        self.detect_proj_frame = tk.Frame(self.dynamic_detect_container)
        self.canny_min_var = tk.IntVar(value=config["CANNY_MIN"])
        tk.Scale(self.detect_proj_frame, variable=self.canny_min_var, from_=0, to=300, orient=tk.HORIZONTAL, label="【プロジェクタ用】エッジ感度 最小値", command=self.on_detect_slider_move).pack(fill=tk.X)
        self.canny_max_var = tk.IntVar(value=config["CANNY_MAX"])
        tk.Scale(self.detect_proj_frame, variable=self.canny_max_var, from_=10, to=500, orient=tk.HORIZONTAL, label="【プロジェクタ用】エッジ感度 最大値", command=self.on_detect_slider_move).pack(fill=tk.X)
        self.proj_thresh_var = tk.IntVar(value=config.get("PROJ_THRESH", 120))
        tk.Scale(self.detect_proj_frame, variable=self.proj_thresh_var, from_=0, to=255, orient=tk.HORIZONTAL, label="【プロジェクタ用】暗い枠の除外 (明度閾値)", command=self.on_detect_slider_move).pack(fill=tk.X)

        self.detect_disp_frame = tk.Frame(self.dynamic_detect_container)
        self.adapt_block_var = tk.IntVar(value=config.get("ADAPTIVE_BLOCK", 21))
        tk.Scale(self.detect_disp_frame, variable=self.adapt_block_var, from_=3, to=99, resolution=2, orient=tk.HORIZONTAL, label="【ディスプレイ用】比較範囲 (奇数)", command=self.on_detect_slider_move).pack(fill=tk.X)
        self.adapt_c_var = tk.DoubleVar(value=config.get("ADAPTIVE_C", 5.0))
        tk.Scale(self.detect_disp_frame, variable=self.adapt_c_var, from_=-20.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL, label="【ディスプレイ用】黒の厳しさ", command=self.on_detect_slider_move).pack(fill=tk.X)

        self.update_slider_visibility()

        tk.Label(target, text="【画質調整】", font=("", 11, "bold")).pack(pady=(10, 2))
        self.black_var = tk.IntVar(value=config["BLACK_LIMIT"])
        tk.Scale(target, variable=self.black_var, from_=0, to=100, orient=tk.HORIZONTAL, label="黒の引き締め", command=self.on_quality_slider_move).pack(fill=tk.X)
        self.gamma_var = tk.DoubleVar(value=config["GAMMA"])
        tk.Scale(target, variable=self.gamma_var, from_=0.5, to=2.5, resolution=0.1, orient=tk.HORIZONTAL, label="ガンマ補正", command=self.on_quality_slider_move).pack(fill=tk.X)
        self.smooth_var = tk.IntVar(value=config["SMOOTH"])
        tk.Scale(target, variable=self.smooth_var, from_=0, to=30, orient=tk.HORIZONTAL, label="ノイズ平滑化", command=self.on_quality_slider_move).pack(fill=tk.X)

    def on_transfer_mode_change(self):
        """通信モードの切り替え時にUIパネルを切り替える"""
        config["TRANSFER_MODE"] = self.transfer_mode_var.get()
        if config["TRANSFER_MODE"] == "FTP":
            self.http_frame.pack_forget()
            self.ftp_frame.pack(fill=tk.X)
        else:
            self.ftp_frame.pack_forget()
            self.http_frame.pack(fill=tk.X)
        save_config()

    def on_aspect_change(self):
        """出力画像の縦横比設定変更時の処理"""
        config["ASPECT_RATIO"] = self.aspect_var.get()
        save_config()

    def on_mode_change(self):
        """検出対象モード（Auto / プロジェクター / ディスプレイ）変更時の処理"""
        self.update_slider_visibility()
        self.on_detect_slider_move()

    def update_slider_visibility(self):
        """選択中のモードに応じて、関連する設定スライダーのみを表示・非表示にする"""
        mode = self.target_mode_var.get()
        if mode == "Auto":
            self.detect_proj_frame.pack(fill=tk.X)
            self.detect_disp_frame.pack(fill=tk.X)
        elif mode == "Projector":
            self.detect_proj_frame.pack(fill=tk.X)
            self.detect_disp_frame.pack_forget()
        elif mode == "Display":
            self.detect_proj_frame.pack_forget()
            self.detect_disp_frame.pack(fill=tk.X)

    def setup_canvas_ui(self):
        """右側のメイン画像表示キャンバスとボタンのUI構築"""
        self.canvas = tk.Canvas(self.right_frame, bg="black", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
        
        btn_frame = tk.Frame(self.right_frame, bg="gray")
        btn_frame.pack(fill=tk.X, pady=5, padx=10)
        
        tk.Button(btn_frame, text="リセット", command=self.reset_crop, width=15).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="確定して再コピー (Enter または Ctrl+C)", command=self.confirm_crop, width=35, bg="lightgreen", font=("", 10, "bold")).pack(side=tk.RIGHT)
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.root.bind("<Return>", lambda e: self.confirm_crop())
        self.root.bind("<Control-c>", lambda e: self.confirm_crop())
        self.root.bind("<Control-C>", lambda e: self.confirm_crop())
        
        self.canvas.create_text(300, 300, text="待機中...\nスマホで写真を撮影してください", fill="white", justify=tk.CENTER, font=("", 14), tags="bg_text")

    def setup_thumbnail_ui(self):
        """右側下部の処理履歴サムネイル表示UIの構築"""
        self.thumb_container = tk.Frame(self.right_frame, height=120, bg="#222")
        self.thumb_container.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        
        self.thumb_canvas = tk.Canvas(self.thumb_container, height=100, bg="#222", highlightthickness=0)
        self.thumb_scrollbar = tk.Scrollbar(self.thumb_container, orient=tk.HORIZONTAL, command=self.thumb_canvas.xview)
        self.thumb_canvas.configure(xscrollcommand=self.thumb_scrollbar.set)
        
        self.thumb_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.thumb_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.thumb_inner_frame = tk.Frame(self.thumb_canvas, bg="#222")
        self.thumb_canvas.create_window((0, 0), window=self.thumb_inner_frame, anchor="nw")
        self.thumb_inner_frame.bind("<Configure>", lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all")))
        self.thumbnail_images = []

    def apply_topmost(self):
        """ウィンドウを常に最前面に表示する設定の適用"""
        is_topmost = self.topmost_var.get()
        self.root.attributes("-topmost", is_topmost)
        config["ALWAYS_ON_TOP"] = is_topmost
        save_config()

    def on_detect_slider_move(self, _=None):
        """枠検出の設定変更時に、少し遅延を入れてから再検出を行う（連続処理を防ぐため）"""
        config["TARGET_MODE"] = self.target_mode_var.get()
        config["AREA_TH"] = self.area_var.get()
        config["APPROX_TH"] = self.approx_var.get()
        config["CANNY_MIN"] = self.canny_min_var.get()
        config["CANNY_MAX"] = self.canny_max_var.get()
        config["PROJ_THRESH"] = self.proj_thresh_var.get()
        config["ADAPTIVE_BLOCK"] = self.adapt_block_var.get()
        config["ADAPTIVE_C"] = self.adapt_c_var.get()
        save_config()
        
        if self._detect_timer is not None:
            self.root.after_cancel(self._detect_timer)
        self._detect_timer = self.root.after(300, self.update_detection_box)

    def update_detection_box(self):
        """現在表示中の画像に対して、変更された設定で再度枠検出を実行して表示を更新する"""
        if self.current_orig_img is not None:
            pts_orig = get_screen_points(self.current_orig_img, config["AREA_TH"], config["APPROX_TH"], config["CANNY_MIN"], config["CANNY_MAX"], config["ADAPTIVE_BLOCK"], config["ADAPTIVE_C"], config.get("PROJ_THRESH", 120), config["TARGET_MODE"])
            self.pts_canvas = pts_orig * self.canvas_scale
            self.initial_pts_canvas = self.pts_canvas.copy()
            self.draw_canvas()

    def on_quality_slider_move(self, _):
        """画質調整のスライダー変更時に、プレビュー更新を遅延実行する"""
        mode = self.target_mode_var.get()
        prefix = "AUTO_" if mode == "Auto" else "PROJ_" if mode == "Projector" else "DISP_"
        config[prefix + "BLACK_LIMIT"] = self.black_var.get()
        config[prefix + "GAMMA"] = self.gamma_var.get()
        config[prefix + "SMOOTH"] = self.smooth_var.get()
        save_config()
        
        if self._preview_timer is not None:
            self.root.after_cancel(self._preview_timer)
        self._preview_timer = self.root.after(200, self.update_preview_image)

    def update_preview_image(self):
        """変更された画質調整設定をもとに、プレビュー用の画像を再生成して表示する"""
        if self.base_disp_img is None: return
        
        img = self.base_disp_img.copy()
        limit = self.black_var.get()
        gamma = self.gamma_var.get()
        smooth = self.smooth_var.get()
        
        gray_wb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dynamic_black = np.percentile(gray_wb, 1)
        if dynamic_black > limit: dynamic_black = limit 
        
        lut = np.zeros((256,), dtype=np.uint8)
        for i in range(256):
            if i < dynamic_black: val = 0
            elif i > 255: val = 255
            else: val = np.clip(255.0 * (i - dynamic_black) / (255 - dynamic_black), 0, 255)
            lut[i] = np.clip(pow(val / 255.0, gamma) * 255.0, 0, 255)
            
        img = cv2.LUT(img, lut)
        if smooth > 0:
            img = cv2.bilateralFilter(img, d=smooth, sigmaColor=75, sigmaSpace=75)
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.draw_canvas()

    def save_settings(self):
        """ネットワークなどの設定を保存する"""
        config["FTP_HOST"] = self.entries["FTP_HOST"].get()
        config["FTP_PORT"] = int(self.entries["FTP_PORT"].get())
        config["FTP_USER"] = self.entries["FTP_USER"].get()
        config["FTP_PASS"] = self.entries["FTP_PASS"].get()
        config["FTP_DIR"] = self.entries["FTP_DIR"].get()
        config["HTTP_PORT"] = int(self.entries["HTTP_PORT"].get())
        save_config()

    def update_gui_loop(self):
        """GUIの定期更新ループ（Ping状態の表示更新や、別スレッドからの画像処理結果の受け取り）"""
        self.status_lbl.config(text=ping_status["status"], fg=ping_status["color"])
        self.ping_lbl.config(text=f"Ping: {ping_status['ping']}")

        # キューに溜まったメッセージをすべて取り出して処理遅延を防ぐ
        while True:
            try:
                item = gui_queue.get_nowait()
                if item["type"] == "new":
                    self.add_thumbnail(item["orig_path"], item["output_path"])
                    self.load_image_to_canvas(item["orig_path"], item["output_path"], item["pts_orig"])
                    
                    # UIへの画像表示と同時に同期的に処理を実行する
                    self.root.update_idletasks()
                    apply_correction_and_save(self.current_orig_img.copy(), item["pts_orig"], item["output_path"])
                    copy_file_to_clipboard(os.path.join(LOCAL_DIR, "temp.jpg"))
                    elapsed = time.time() - item.get("start_time", time.time())
                    self.process_status_lbl.config(text=f"✔ 処理完了 (コピー済) - {elapsed:.2f}秒", fg="green")
                elif item["type"] == "status":
                    self.process_status_lbl.config(text=item["message"], fg=item.get("color", "black"))
            except queue.Empty:
                break

        self.root.after(100, self.update_gui_loop)

    def add_thumbnail(self, orig_path, output_path):
        """処理が完了した画像をサムネイルとして下部リストに追加する"""
        try:
            img = Image.open(orig_path)
            img.thumbnail((120, 100))
            photo = ImageTk.PhotoImage(img)
            self.thumbnail_images.append(photo)
            
            btn = tk.Button(self.thumb_inner_frame, image=photo, relief=tk.FLAT, bg="#222", 
                            command=lambda: self.on_thumbnail_click(orig_path, output_path))
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            
            self.thumb_canvas.update_idletasks()
            self.thumb_canvas.xview_moveto(1.0)
        except Exception as e:
            pass

    def on_thumbnail_click(self, orig_path, output_path):
        """サムネイルクリック時に、該当画像をメインキャンバスに再読み込みする"""
        img = cv2.imread(orig_path)
        if img is None: return
        pts_orig = get_screen_points(img, config["AREA_TH"], config["APPROX_TH"], config["CANNY_MIN"], config["CANNY_MAX"], config["ADAPTIVE_BLOCK"], config["ADAPTIVE_C"], config.get("PROJ_THRESH", 120), config["TARGET_MODE"])
        self.load_image_to_canvas(orig_path, output_path, pts_orig)

    def load_image_to_canvas(self, orig_path, output_path, pts_orig):
        """画像をキャンバスに表示できるようにリサイズし、状態を初期化する"""
        orig = cv2.imread(orig_path)
        if orig is None: return
        
        self.current_orig_img = orig
        self.current_output_path = output_path
        
        h, w = orig.shape[:2]
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 100: canvas_w, canvas_h = 560, 500
        
        self.canvas_scale = min(canvas_w / w, canvas_h / h)
        disp_w, disp_h = int(w * self.canvas_scale), int(h * self.canvas_scale)
        
        self.base_disp_img = cv2.resize(orig, (disp_w, disp_h))
        self.pts_canvas = pts_orig * self.canvas_scale
        self.initial_pts_canvas = self.pts_canvas.copy()
        
        self.snap_edges = None
        self.snap_corners = []
        
        self.update_preview_image()
        self.generate_snap_points_async()

    def generate_snap_points_async(self):
        """スナップ用の特徴点（角・辺）画像の生成を裏で実行する"""
        if self.base_disp_img is None: return
        img_copy = self.base_disp_img.copy()

        def generate():
            gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.02, minDistance=10)
            
            self.snap_edges = edges
            self.snap_corners = corners.reshape(-1, 2) if corners is not None else []
            
        threading.Thread(target=generate, daemon=True).start()

    def get_snap_point(self, x, y, radius=20):
        """指定座標の周辺から、画像の角や辺を探してスナップ（吸着）する座標を返す"""
        h, w = self.base_disp_img.shape[:2]
        best_x, best_y = x, y
        min_dist = radius ** 2
        is_snapped = False

        # 1. 画像から検出した「角（コーナー）」へのスナップを最優先
        if hasattr(self, 'snap_corners') and len(self.snap_corners) > 0:
            dists = np.sum((self.snap_corners - [x, y]) ** 2, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < min_dist:
                best_x, best_y = self.snap_corners[idx]
                min_dist = dists[idx]
                is_snapped = True

        # 2. コーナーがなければ「辺（エッジ）」へのスナップ
        if not is_snapped and hasattr(self, 'snap_edges') and self.snap_edges is not None:
            x1, y1 = max(0, int(x - radius)), max(0, int(y - radius))
            x2, y2 = min(w, int(x + radius + 1)), min(h, int(y + radius + 1))
            if x2 > x1 and y2 > y1:
                roi = self.snap_edges[y1:y2, x1:x2]
                pts = np.argwhere(roi > 0)
                if len(pts) > 0:
                    cy, cx = y - y1, x - x1
                    dists = (pts[:, 0] - cy)**2 + (pts[:, 1] - cx)**2
                    idx = np.argmin(dists)
                    best_y = y1 + pts[idx][0]
                    best_x = x1 + pts[idx][1]
                    is_snapped = True

        # 3. エッジもなければ「画像の物理的な境界線」へのスナップ
        if not is_snapped:
            if x < radius: best_x = 0; is_snapped = True
            elif x > w - radius: best_x = w; is_snapped = True
            if y < radius: best_y = 0; is_snapped = True
            elif y > h - radius: best_y = h; is_snapped = True

        return best_x, best_y, is_snapped

    def draw_canvas(self):
        """キャンバス上に画像と、現在設定されている四角形（ポリゴンと頂点）を描画する"""
        self.canvas.delete("all")
        if hasattr(self, 'tk_image'):
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
            pts = self.pts_canvas.flatten().tolist()
            self.canvas.create_polygon(pts, outline='green', fill='', width=2)
            
            r = 6
            for x, y in self.pts_canvas:
                self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='red', outline='white')

            # スナップ中の強調表示（黄色い二重リングと文字）
            for sx, sy in self.current_snap_points:
                self.canvas.create_oval(sx-12, sy-12, sx+12, sy+12, outline='yellow', width=2)
                self.canvas.create_text(sx + 15, sy - 15, text="スナップ", fill="yellow", anchor=tk.W, font=("", 10, "bold"))

    def on_press(self, event):
        """マウス押下イベント。クリック位置が頂点か辺に近いかを判定し、ドラッグモードに移行する"""
        if self.current_orig_img is None: return
        p = np.array([event.x, event.y])
        self.current_snap_points = []
        
        # まず頂点の付近(半径20px以内)かどうか判定
        dists = [np.linalg.norm(self.pts_canvas[i] - p) for i in range(4)]
        min_idx = np.argmin(dists)
        if dists[min_idx] < 20: 
            self.drag_mode = "point"
            self.drag_idx = min_idx
            return

        def point_to_segment(p, p1, p2):
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0: return np.linalg.norm(p - p1)
            t = max(0, min(1, np.dot(p - p1, p2 - p1) / l2))
            projection = p1 + t * (p2 - p1)
            return np.linalg.norm(p - projection)
            
        # 頂点でなければ辺の付近かどうか判定
        edge_dists = [point_to_segment(p, self.pts_canvas[i], self.pts_canvas[(i+1)%4]) for i in range(4)]
        min_edge_idx = np.argmin(edge_dists)
        if edge_dists[min_edge_idx] < 20:
            self.drag_mode = "edge"
            self.drag_idx = min_edge_idx
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.drag_start_pts = self.pts_canvas.copy()

    def on_drag(self, event):
        """マウスドラッグイベント。選択中の頂点や辺を移動させる"""
        if self.current_orig_img is None: return
        
        h, w = self.base_disp_img.shape[:2]
        x = max(0, min(w, event.x))
        y = max(0, min(h, event.y))

        if self.drag_mode == "point":
            snap_x, snap_y, is_snapped = self.get_snap_point(x, y, radius=20)
            self.pts_canvas[self.drag_idx] = [snap_x, snap_y]
            self.current_snap_points = [(snap_x, snap_y)] if is_snapped else []
            self.draw_canvas()
            
        elif self.drag_mode == "edge":
            dx = x - self.drag_start_x
            dy = y - self.drag_start_y
            
            p1_idx = self.drag_idx
            p2_idx = (self.drag_idx + 1) % 4
            
            ideal_p1_x = self.drag_start_pts[p1_idx][0] + dx
            ideal_p1_y = self.drag_start_pts[p1_idx][1] + dy
            ideal_p2_x = self.drag_start_pts[p2_idx][0] + dx
            ideal_p2_y = self.drag_start_pts[p2_idx][1] + dy
            
            snap_p1_x, snap_p1_y, snap_1 = self.get_snap_point(ideal_p1_x, ideal_p1_y, radius=20)
            snap_p2_x, snap_p2_y, snap_2 = self.get_snap_point(ideal_p2_x, ideal_p2_y, radius=20)

            self.pts_canvas[p1_idx] = [snap_p1_x, snap_p1_y]
            self.pts_canvas[p2_idx] = [snap_p2_x, snap_p2_y]
            
            snaps = []
            if snap_1: snaps.append((snap_p1_x, snap_p1_y))
            if snap_2: snaps.append((snap_p2_x, snap_p2_y))
            self.current_snap_points = snaps
            self.draw_canvas()

    def on_release(self, event):
        """マウス離上イベント。ドラッグ状態を解除する"""
        self.drag_mode = None
        self.drag_idx = -1
        self.current_snap_points = []
        self.draw_canvas()

    def reset_crop(self):
        """枠の形状を初期検出時の状態にリセットする"""
        if self.current_orig_img is not None:
            self.pts_canvas = self.initial_pts_canvas.copy()
            self.draw_canvas()

    def confirm_crop(self):
        """現在の枠設定で最終的な補正・保存・クリップボードへのコピーを実行する"""
        if self.current_orig_img is None: return
        
        pts_orig = self.pts_canvas / self.canvas_scale
        output_path = self.current_output_path
        
        self.process_status_lbl.config(text="⚙ 手動処理中...", fg="orange")
        self.root.update_idletasks()
        start_time = time.time()
        
        apply_correction_and_save(self.current_orig_img, pts_orig, output_path)
        copy_file_to_clipboard(os.path.join(LOCAL_DIR, "temp.jpg"))
        elapsed = time.time() - start_time
        
        self.canvas.delete("status_msg")
        self.canvas.create_text(
            10, 10, anchor=tk.NW, 
            text=f"✔ 確定・コピー完了 ({elapsed:.2f}秒)", 
            fill="black", font=("", 14, "bold"), tags="status_msg"
        )
        self.canvas.create_text(
            12, 12, anchor=tk.NW, 
            text=f"✔ 確定・コピー完了 ({elapsed:.2f}秒)", 
            fill="lightgreen", font=("", 14, "bold"), tags="status_msg"
        )
        
        self.process_status_lbl.config(text=f"✔ 手動処理完了 (コピー済) - {elapsed:.2f}秒", fg="green")

if __name__ == "__main__":
    load_config()
    root = tk.Tk()
    app = ScannerApp(root)
    root.mainloop()