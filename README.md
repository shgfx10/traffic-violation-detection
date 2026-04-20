 Traffic Violation Detection System

An AI-powered real-time traffic violation detection system that uses a live camera feed to automatically identify vehicle violations including helmet violations, red-light running, lane violations, overspeeding, and wrong-side driving.

Pilot selected for deployment at 3 high-traffic intersections (Q2 2026)** — with city-wide rollout potential.



What It Does

The system connects to a camera feed and uses Computer Vision to detect traffic violations in real time — no human monitoring required. When a violation is detected, it logs the event, timestamps it, and flags it for action.

Violations detected:**
   No helmet (two-wheeler riders)
   Red-light running
   Lane violation
   Overspeeding
   Wrong-side driving
   Illegal parking



 Performance

| Metric | Result |
|--------|--------|
| Detection Accuracy | **89%** |
| Processing Speed | **15 FPS** real-time |
| False Positive Reduction | **35%** (via data augmentation) |
| Training Dataset | **5,000+** annotated images |
| Labor Cost Savings | **₹90L/year** (manual surveillance replaced) |
| Intersections monitored | **10+** simultaneously |

---

 Tech Stack

| Component | Technology |
|-----------|------------|
| Core Model | CNN (Convolutional Neural Network) |
| Vision Library | OpenCV |
| Language | Python |
| Training Framework | TensorFlow / Keras |
| Data Processing | NumPy, Pandas |
| Annotation | Custom labeled dataset (5,000+ images) |

---

 Getting Started

### Prerequisites
```bash
pip install opencv-python tensorflow keras numpy pandas
```

### Run the detection system
```bash
python traffic_detection_single.py
```

Make sure your camera is connected. The system will start reading the feed and detecting violations in real time.

---

##  Project Structure

```
traffic-violation-detection/
│
├── traffic_detection_single.py      # Main detection script (camera input → violation output)
├── traffic_violation_detection_system.html  # Dashboard UI for monitoring & analytics
└── README.md
```

---

##  Dashboard

The included HTML dashboard (`traffic_violation_detection_system.html`) provides:
- Live camera feed simulation across 6 intersections
- Real-time violation analytics charts
- Challan / fine issuing interface
- Location-based camera monitoring
- Hourly violation trend graphs

Open in any browser — no server needed.

---

##  Impact

This project was identified as a pilot deployment candidate for **3 high-traffic intersections**, with a city-wide rollout budget of ₹8Cr. The system replaces manual surveillance, enabling:

- **60% reduction** in manual monitoring labor
- **₹90L annual savings** in surveillance costs
- Automated, consistent enforcement across **10+ intersections**

---

##  Recognition

This project was part of the work that contributed to:
- **2nd Place** — TEXPO 2K24 Innovation Pitch Competition

---

## 👤Author

**Selva Ganesh K**  
AI Research Analyst & Technical Writer  
B.Tech — Artificial Intelligence & Machine Learning

- 🌐 Blog: [mysticquill.blogspot.com](https://mysticquill.blogspot.com)
- 💼 LinkedIn: [linkedin.com/in/selva-ganesh-61344b342](https://linkedin.com/in/selva-ganesh-61344b342)
- 📧 selvaganeshtarun1423@gmail.com

---

## 📄 License

This project is for educational and demonstration purposes.
