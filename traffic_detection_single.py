"""
Traffic Violation Detection System - Single File Version
Run: python traffic_detection_single.py
"""

import cv2
import numpy as np
import sqlite3
import time
import os
from datetime import datetime
from collections import OrderedDict

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SPEED_LIMIT     = 60
PIXELS_PER_METER = 8.5
MIN_AREA        = 2000
MAX_AREA        = 150000

# ─── DATABASE ─────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("violations.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, plate TEXT, speed INTEGER,
        confidence REAL, timestamp TEXT, camera TEXT
    )""")
    conn.commit()
    return conn

def save_violation(conn, v):
    conn.execute("INSERT INTO violations (type,plate,speed,confidence,timestamp,camera) VALUES (?,?,?,?,?,?)",
        (v['type'], v.get('plate','Unknown'), v.get('speed',0),
         v.get('confidence',0), datetime.now().isoformat(), "CAM-01"))
    conn.commit()

def print_report(conn):
    total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
    print(f"\n── Violation Report ──────────────────")
    print(f"  Total violations: {total}")
    rows = conn.execute("SELECT type, COUNT(*) FROM violations GROUP BY type").fetchall()
    for r in rows:
        print(f"  {r[0]:25s}: {r[1]}")
    print("──────────────────────────────────────")

# ─── TRACKER ──────────────────────────────────────────────────────────────────
class Tracker:
    def __init__(self):
        self.objects     = OrderedDict()
        self.disappeared = OrderedDict()
        self.next_id     = 1
        self.speed_hist  = {}

    def update(self, detections):
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > 30:
                    del self.objects[oid]
                    del self.disappeared[oid]
            return list(self.objects.values())

        centroids = [d['centroid'] for d in detections]

        if not self.objects:
            for d in detections:
                self._register(d)
        else:
            oids = list(self.objects.keys())
            ocents = [v['centroid'] for v in self.objects.values()]
            D = self._dist(ocents, centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_r, used_c = set(), set()
            for r, c in zip(rows, cols):
                if r in used_r or c in used_c or D[r,c] > 80:
                    continue
                oid = oids[r]
                self.objects[oid].update({'centroid': centroids[c], 'bbox': detections[c]['bbox']})
                self.disappeared[oid] = 0
                used_r.add(r); used_c.add(c)
            for r in set(range(len(oids))) - used_r:
                oid = oids[r]
                self.disappeared[oid] = self.disappeared.get(oid, 0) + 1
                if self.disappeared[oid] > 30:
                    del self.objects[oid]; del self.disappeared[oid]
            for c in set(range(len(detections))) - used_c:
                self._register(detections[c])

        return list(self.objects.values())

    def _register(self, d):
        import random, string
        plate = f"TN{random.randint(1,99):02d} {''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ',k=2))} {random.randint(1000,9999)}"
        self.objects[self.next_id] = {
            'id': self.next_id, 'centroid': d['centroid'],
            'bbox': d['bbox'], 'plate': plate, 'trail': [d['centroid']]
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def estimate_speed(self, vid, cy):
        now = time.time()
        if vid not in self.speed_hist:
            self.speed_hist[vid] = []
        self.speed_hist[vid].append((now, cy))
        self.speed_hist[vid] = [p for p in self.speed_hist[vid] if now - p[0] < 2.0]
        h = self.speed_hist[vid]
        if len(h) < 2:
            return 0
        dt = h[-1][0] - h[0][0]
        dy = abs(h[-1][1] - h[0][1])
        if dt <= 0: return 0
        return round((dy / PIXELS_PER_METER / dt) * 3.6, 1)

    @staticmethod
    def _dist(a, b):
        a = np.array(a, float); b = np.array(b, float)
        return np.sqrt(np.maximum(
            np.sum(a**2,axis=1,keepdims=True) + np.sum(b**2,axis=1) - 2*np.dot(a,b.T), 0))

# ─── DRAW ─────────────────────────────────────────────────────────────────────
def draw(frame, vehicles, violations, stats):
    viol_ids = {v.get('vehicle_id') for v in violations}
    h, w = frame.shape[:2]

    for v in vehicles:
        x, y, bw, bh = v.get('bbox', (0,0,0,0))
        vid = v['id']
        color = (0,60,255) if vid in viol_ids else (0,200,80)
        cv2.rectangle(frame, (x,y), (x+bw,y+bh), color, 2)
        label = f"V{vid} | {v.get('plate','???')}"
        cv2.rectangle(frame, (x, y-20), (x+len(label)*8, y), color, -1)
        cv2.putText(frame, label, (x+2, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1)

        trail = v.get('trail', [])
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (0,160,60), 1)

    for i, vl in enumerate(violations):
        bx, by = w-240, 40 + i*46
        if by+40 > h: break
        cv2.rectangle(frame, (bx,by), (bx+232,by+40), (0,0,0), -1)
        cv2.rectangle(frame, (bx,by), (bx+232,by+40), (0,60,255), 1)
        cv2.putText(frame, f"! {vl.get('type','Violation')}", (bx+5,by+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,60,255), 1)
        cv2.putText(frame, f"{vl.get('plate','???')}  {vl.get('confidence',0)*100:.0f}%",
                    (bx+5,by+32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # HUD
    cv2.rectangle(frame, (0,0), (w,32), (0,0,0), -1)
    ts = datetime.now().strftime("%H:%M:%S")
    hud = f"TVDS | Vehicles:{stats['vehicles']} | Violations:{stats['total']} | FPS:{stats['fps']:.1f} | {ts}"
    cv2.putText(frame, hud, (6,22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,220,100), 1)
    return frame

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  TRAFFIC VIOLATION DETECTION SYSTEM")
    print("  Press Q to quit | S to screenshot")
    print("=" * 55)

    conn    = init_db()
    tracker = Tracker()
    bg_sub  = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Try webcam first, then demo mode
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[INFO] No webcam found. Running in DEMO mode with synthetic video.")
        demo_mode = True
    else:
        demo_mode = False
        print("[INFO] Webcam opened successfully!")

    total_viols  = 0
    vehicle_count = 0
    frame_no     = 0
    start        = time.time()

    while True:
        if demo_mode:
            # Create a synthetic demo frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (20, 25, 35)
            # Draw road
            pts = np.array([[220,480],[420,480],[380,200],[260,200]], np.int32)
            cv2.fillPoly(frame, [pts], (40,50,65))
            cv2.line(frame, (320,480), (320,200), (60,70,80), 2)
            # Simulate moving vehicles
            t = frame_no / 30.0
            for i in range(3):
                vx = int(280 + i*40 + 10*np.sin(t+i))
                vy = int(100 + ((frame_no*2 + i*160) % 380))
                cv2.rectangle(frame, (vx-20,vy-15), (vx+20,vy+15), (60,80,120), -1)
                cv2.rectangle(frame, (vx-16,vy-8), (vx+16,vy-2), (100,140,180), -1)
            ret = True
        else:
            ret, frame = cap.read()

        if not ret:
            break

        frame_no += 1

        # Background subtraction
        fg = bg_sub.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh)
            if aspect < 0.3 or aspect > 6:
                continue
            cx, cy = x + bw//2, y + bh//2
            detections.append({'bbox':(x,y,bw,bh), 'centroid':(cx,cy)})

        tracked = tracker.update(detections)

        # Update trails
        for v in tracked:
            trail = v.get('trail', [])
            trail.append(v['centroid'])
            if len(trail) > 20: trail.pop(0)
            v['trail'] = trail
            vehicle_count = max(vehicle_count, v['id'])

        # Check violations
        violations = []
        for v in tracked:
            _, cy = v['centroid']
            speed = tracker.estimate_speed(v['id'], cy)
            v['speed'] = speed
            if speed > SPEED_LIMIT:
                viol = {
                    'type': 'Overspeeding',
                    'vehicle_id': v['id'],
                    'plate': v.get('plate','Unknown'),
                    'speed': int(speed),
                    'confidence': min(0.99, 0.7 + (speed-SPEED_LIMIT)/200),
                }
                violations.append(viol)
                save_violation(conn, viol)
                total_viols += 1
                print(f"[VIOLATION] Overspeed | {viol['plate']} | {int(speed)} km/h")

        fps = frame_no / max(time.time()-start, 0.001)
        stats = {'vehicles': vehicle_count, 'total': total_viols, 'fps': fps}
        frame = draw(frame, tracked, violations, stats)

        cv2.imshow("Traffic Violation Detection System - Press Q to quit", frame)

        key = cv2.waitKey(1 if not demo_mode else 33) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fn = f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[INFO] Saved: {fn}")

    if not demo_mode:
        cap.release()
    cv2.destroyAllWindows()
    print_report(conn)
    conn.close()

if __name__ == "__main__":
    main()
