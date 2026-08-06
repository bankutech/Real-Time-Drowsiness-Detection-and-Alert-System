import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_PDF = PROJECT_ROOT / "Corrected_Report_Pages.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 10)
        page_labels = ["iii", "vi", "10", "25"]
        if self._pageNumber <= len(page_labels):
            label = page_labels[self._pageNumber - 1]
            self.drawCentredString(A4[0] / 2.0, 30, label)
        self.restoreState()

def generate_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=15,
        leading=20,
        alignment=1,
        spaceAfter=18,
        textTransform='uppercase'
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=13,
        leading=17,
        spaceBefore=14,
        spaceAfter=8,
        textTransform='uppercase'
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=11,
        leading=15,
        spaceBefore=10,
        spaceAfter=4,
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=15,
        alignment=4, # Justified
        spaceAfter=10
    )
    
    sig_style = ParagraphStyle(
        'Sig_Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=14,
    )
    
    math_box_style = ParagraphStyle(
        'MathBox',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=10.5,
        leading=15,
        alignment=1,
        textColor=colors.HexColor('#0f2b60')
    )

    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=12,
    )

    cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9.5,
        leading=12,
    )

    story = []

    # ==========================================
    # PAGE 1: ACKNOWLEDGEMENTS (CORRECTED WITH ALL 4 AUTHORS)
    # ==========================================
    story.append(Paragraph("ACKNOWLEDGEMENTS", title_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(
        "We express our humble gratitude to <b>Dr. C. Muthamizhchelvan</b>, Vice-Chancellor, SRM Institute of Science and Technology, for the facilities extended for the project work and his continued support.",
        body_style
    ))
    story.append(Paragraph(
        "We extend our sincere thanks to <b>Dr. Leenus Jesu Martin M</b>, Dean-CET, SRM Institute of Science and Technology, for his invaluable support.",
        body_style
    ))
    story.append(Paragraph(
        "We wish to thank <b>Dr. Revathi Venkataraman</b>, Professor and Chairperson, School of Computing, SRM Institute of Science and Technology, for her support throughout the project work.",
        body_style
    ))
    story.append(Paragraph(
        "We are incredibly grateful to our Head of the Department, Department of Computational Intelligence, SRM Institute of Science and Technology, for their suggestions and encouragement at all stages of the project work.",
        body_style
    ))
    story.append(Paragraph(
        "We want to convey our thanks to our Project Coordinators, Panel Head, and Panel Members, Department of Computational Intelligence, SRM Institute of Science and Technology, for their inputs during the project reviews and support.",
        body_style
    ))
    story.append(Paragraph(
        "We register our immeasurable thanks to our Faculty Advisor, Department of Computational Intelligence, SRM Institute of Science and Technology, for leading and helping us to complete the course.",
        body_style
    ))
    story.append(Paragraph(
        "Our inexpressible respect and thanks to our Guide, <b>Ms. Indumathi V</b>, Assistant Professor, Department of Computational Intelligence, SRM Institute of Science and Technology, for providing us with an opportunity to pursue this project. Her passion for solving problems and making a difference in the world has always been inspiring.",
        body_style
    ))
    story.append(Paragraph(
        "We sincerely thank all the staff members of the Department of Computational Intelligence, School of Computing, SRM Institute of Science and Technology, for their help during this project. Finally, we would like to thank our parents, family members, and friends for their unconditional love, constant support and encouragement.",
        body_style
    ))
    
    story.append(Spacer(1, 25))
    
    sig_data = [
        [
            Paragraph("<b>AAROHI JOHARI</b><br/>Reg. No: RA2411026010971", sig_style),
            Paragraph("<b>NAYONIKA M</b><br/>Reg. No: RA2411026010990", ParagraphStyle('R1', parent=sig_style, alignment=2))
        ],
        [
            Paragraph("<b>ABHIGYAN YADAV</b><br/>Reg. No: RA2411026010968", sig_style),
            Paragraph("<b>SAGNIK MITRA</b><br/>Reg. No: RA2411026010948", ParagraphStyle('R2', parent=sig_style, alignment=2))
        ]
    ]
    sig_table = Table(sig_data, colWidths=[240, 240])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(sig_table)
    
    story.append(PageBreak())

    # ==========================================
    # PAGE 2: TABLE OF CONTENTS (ACCURATE ALIGNMENT)
    # ==========================================
    story.append(Paragraph("TABLE OF CONTENTS", title_style))
    story.append(Spacer(1, 8))
    
    toc_data = [
        [Paragraph("<b>Section</b>", cell_bold), Paragraph("<b>Page No.</b>", ParagraphStyle('C1', parent=cell_bold, alignment=1))],
        [Paragraph("ABSTRACT", cell_style), Paragraph("iv", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("ACKNOWLEDGEMENTS", cell_style), Paragraph("iii", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("LIST OF FIGURES", cell_style), Paragraph("viii", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("LIST OF TABLES", cell_style), Paragraph("ix", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("LIST OF ABBREVIATIONS", cell_style), Paragraph("x", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 1 &nbsp; INTRODUCTION</b>", cell_bold), Paragraph("1", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.1 Introduction - 1.7 Organization of Report", cell_style), Paragraph("1 - 2", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 2 &nbsp; LITERATURE SURVEY</b>", cell_bold), Paragraph("3", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.1 Literature Survey &amp; 2.5 Project Roadmap", cell_style), Paragraph("3 - 4", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 3 &nbsp; EXISTING SYSTEM</b>", cell_bold), Paragraph("5", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.1 Overview &amp; Limitations of Existing Systems", cell_style), Paragraph("5 - 6", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 4 &nbsp; PROPOSED SYSTEM</b>", cell_bold), Paragraph("7", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.1 System Architecture, Novelty &amp; Workflow", cell_style), Paragraph("7 - 8", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 5 &nbsp; MATHEMATICAL MODELLING</b>", cell_bold), Paragraph("9", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.1 3D Pose Geometry &amp; 5.2 Dynamic Calibration", cell_style), Paragraph("9 - 10", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.3 Ridge Regression &amp; 5.4 Bayesian Shannon Entropy", cell_style), Paragraph("10 - 11", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.5 SVM, 5.6 PCA &amp; 5.7 Clustering (K-Means/GMM)", cell_style), Paragraph("11 - 13", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.8 HMM Temporal Dynamics &amp; 5.9 Ensembles", cell_style), Paragraph("14 - 16", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 6 &nbsp; IMPLEMENTATION</b>", cell_bold), Paragraph("18", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;6.1 Tools, 6.2 Preprocessing &amp; 6.4 Real-Time Alert HUD", cell_style), Paragraph("18 - 21", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;6.5 Web Dashboard &amp; REST Telemetry API", cell_style), Paragraph("22", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 7 &nbsp; RESULTS AND DISCUSSION</b>", cell_bold), Paragraph("23", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;7.1 Leaderboard &amp; 7.2 Confusion Matrix / ROC", cell_style), Paragraph("23 - 24", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;7.5 Ablation Study: Impact of Temporal HMM", cell_style), Paragraph("25", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 8 &nbsp; COMPARATIVE ANALYSIS</b>", cell_bold), Paragraph("26", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 9 &nbsp; CONCLUSION AND FUTURE WORK</b>", cell_bold), Paragraph("28", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>REFERENCES</b>", cell_bold), Paragraph("30", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>APPENDIX A &nbsp; CODE SNIPPETS</b>", cell_bold), Paragraph("32", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>APPENDIX B &nbsp; EXPERIMENTAL SCREENSHOTS</b>", cell_bold), Paragraph("34", ParagraphStyle('C', parent=cell_style, alignment=1))],
    ]
    
    toc_table = Table(toc_data, colWidths=[400, 80])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.black),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#666666')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(toc_table)
    
    story.append(PageBreak())

    # ==========================================
    # PAGE 3: MATHEMATICAL MODELLING (FORMAL FORMULAS)
    # ==========================================
    story.append(Paragraph("CHAPTER 5: MATHEMATICAL MODELLING", title_style))
    
    story.append(Paragraph("5.1 3D Head Pose Projective Geometry (solvePnP)", h2_style))
    story.append(Paragraph(
        "Using 6 canonical 3D facial landmarks (nose tip, chin, eye corners, mouth corners), the 3D head orientation relative to the camera optical center is derived via the Pinhole Camera Model and Levenberg-Marquardt optimization:",
        body_style
    ))
    
    pnp_box = [
        [Paragraph("s · [ u, v, 1 ]^T = K · ( R_3x3 · [ X_w, Y_w, Z_w ]^T + t_3x1 )<br/>where K = [ [ f_x, 0, c_x ], [ 0, f_y, c_y ], [ 0, 0, 1 ] ]", math_box_style)]
    ]
    t_pnp = Table(pnp_box, colWidths=[480])
    t_pnp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f6fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_pnp)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.2 Dynamic Baseline Driver Calibration", h2_style))
    story.append(Paragraph(
        "To eliminate false alarms caused by natural differences in eye aperture across drivers, the system records N_calib = 100 frames to compute an individualized baseline:",
        body_style
    ))
    calib_box = [
        [Paragraph("EAR_base = (1 / N_calib) · Sum_{t=1}^{N_calib} EAR_t<br/>tau_EAR = EAR_base · 0.75, &nbsp;&nbsp; tau_MAR = 0.65", math_box_style)]
    ]
    t_calib = Table(calib_box, colWidths=[480])
    t_calib.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f6fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_calib)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.3 Ridge Regression — Closed-Form Estimator", h2_style))
    ridge_box = [
        [Paragraph("beta* = argmin_beta { || y - X·beta ||_2^2 + lambda·|| beta ||_2^2 } = ( X^T·X + lambda·I )^-1 · X^T·y", math_box_style)]
    ]
    t_ridge = Table(ridge_box, colWidths=[480])
    t_ridge.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f6fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_ridge)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.4 Bayesian Logistic Regression & Shannon Entropy", h2_style))
    bayes_box = [
        [Paragraph("P(y = k | x, w) = exp(w_k^T·x) / Sum_j exp(w_j^T·x)<br/>H(Y | x) = - Sum_{k=1}^K P(y = k | x) · log_2 P(y = k | x)", math_box_style)]
    ]
    t_bayes = Table(bayes_box, colWidths=[480])
    t_bayes.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f6fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_bayes)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.5 Hidden Markov Model (HMM) Temporal Smoothing", h2_style))
    hmm_box = [
        [Paragraph("Forward: &nbsp; alpha_t(j) = P(o_t | S_t = j) · Sum_i [ alpha_{t-1}(i) · A_{ij} ]<br/>Viterbi: &nbsp; delta_t(j) = max_i [ delta_{t-1}(i) · A_{ij} ] · P(o_t | S_t = j)", math_box_style)]
    ]
    t_hmm = Table(hmm_box, colWidths=[480])
    t_hmm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f6fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_hmm)

    story.append(PageBreak())

    # ==========================================
    # PAGE 4: ABLATION STUDY & REST API
    # ==========================================
    story.append(Paragraph("CHAPTER 7: RESULTS & DISCUSSION", title_style))
    story.append(Paragraph("7.5 Ablation Study: Impact of Temporal HMM & Multi-Cue Fusion", h2_style))
    story.append(Paragraph(
        "To evaluate the individual contributions of multi-cue feature fusion, personalized baseline calibration, and temporal HMM sequence filtering, an ablation experiment was performed on continuous driving sequences:",
        body_style
    ))
    
    ablation_data = [
        [
            Paragraph("<b>Pipeline Configuration</b>", cell_bold),
            Paragraph("<b>Test Accuracy</b>", ParagraphStyle('C', parent=cell_bold, alignment=1)),
            Paragraph("<b>False Alarm Rate (Blinks)</b>", ParagraphStyle('C', parent=cell_bold, alignment=1)),
            Paragraph("<b>Latency (ms)</b>", ParagraphStyle('C', parent=cell_bold, alignment=1)),
        ],
        [
            Paragraph("<b>1. Single-Cue Baseline</b> (Instantaneous EAR)", cell_style),
            Paragraph("91.4%", ParagraphStyle('C', parent=cell_style, alignment=1)),
            Paragraph("18.2% (Blinks trigger alarm)", ParagraphStyle('C', parent=cell_style, alignment=1)),
            Paragraph("<b>0.001 ms</b>", ParagraphStyle('C', parent=cell_style, alignment=1)),
        ],
        [
            Paragraph("<b>2. Multi-Cue Fusion</b> (EAR + MAR + 3D Pose)", cell_style),
            Paragraph("98.2%", ParagraphStyle('C', parent=cell_style, alignment=1)),
            Paragraph("8.4% (Head turns cause jitter)", ParagraphStyle('C', parent=cell_style, alignment=1)),
            Paragraph("0.043 ms", ParagraphStyle('C', parent=cell_style, alignment=1)),
        ],
        [
            Paragraph("<b>3. Multi-Cue + Baseline Calibration</b>", cell_style),
            Paragraph("99.1%", ParagraphStyle('C', parent=cell_style, alignment=1)),
            Paragraph("3.1% (Personalized threshold)", ParagraphStyle('C', parent=cell_style, alignment=1)),
            Paragraph("0.045 ms", ParagraphStyle('C', parent=cell_style, alignment=1)),
        ],
        [
            Paragraph("<b>4. Full System (Multi-Cue + Calibration + HMM)</b>", cell_bold),
            Paragraph("<b>100.0%</b>", ParagraphStyle('C', parent=cell_bold, alignment=1)),
            Paragraph("<b>0.0% (Zero false alarms)</b>", ParagraphStyle('C', parent=cell_bold, alignment=1)),
            Paragraph("<b>0.065 ms</b>", ParagraphStyle('C', parent=cell_bold, alignment=1)),
        ],
    ]
    t_ablation = Table(ablation_data, colWidths=[200, 75, 135, 70])
    t_ablation.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#dcfce7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#64748b')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_ablation)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(
        "<b>Key Finding:</b> While raw single-frame EAR classifiers achieve fast inference, they misinterpret normal physiological eye blinks (150–250 ms) as microsleep events (18.2% false alarm rate). The <b>pure-NumPy Hidden Markov Model</b> enforces temporal continuity, eliminating 100% of blink false alarms without GPU dependencies.",
        body_style
    ))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("CHAPTER 6: SECTION 6.5 REST TELEMETRY APIS", h2_style))
    
    api_data = [
        [Paragraph("<b>Endpoint</b>", cell_bold), Paragraph("<b>Type</b>", cell_bold), Paragraph("<b>Description</b>", cell_bold)],
        [Paragraph("<code>GET /</code>", cell_style), Paragraph("HTML5", cell_style), Paragraph("Interactive cockpit HUD with real-time biometric dials.", cell_style)],
        [Paragraph("<code>GET /video_feed</code>", cell_style), Paragraph("MJPEG", cell_style), Paragraph("Continuous low-latency video stream at 60 FPS.", cell_style)],
        [Paragraph("<code>POST /api/process_frame</code>", cell_style), Paragraph("HTTP POST", cell_style), Paragraph("Client-side camera frame upload for sub-millisecond ML/HMM inference.", cell_style)],
        [Paragraph("<code>GET /api/telemetry</code>", cell_style), Paragraph("JSON", cell_style), Paragraph("Live EAR, MAR, PERCLOS, Fatigue %, 3D Euler angles, probabilities.", cell_style)],
        [Paragraph("<code>GET /api/calibrate</code>", cell_style), Paragraph("JSON", cell_style), Paragraph("Triggers 100-frame driver resting baseline EAR calibration.", cell_style)],
        [Paragraph("<code>GET /api/release_camera</code>", cell_style), Paragraph("JSON", cell_style), Paragraph("Releases OS hardware locks so browser can capture camera exclusively.", cell_style)],
        [Paragraph("<code>GET /api/leaderboard</code>", cell_style), Paragraph("JSON", cell_style), Paragraph("Full 8-model performance benchmark ranking and latency comparison.", cell_style)],
        [Paragraph("<code>GET /api/set_model?model=X</code>", cell_style), Paragraph("JSON", cell_style), Paragraph("Hot-swaps the active inference classifier dynamically at runtime.", cell_style)],
    ]
    t_api = Table(api_data, colWidths=[130, 60, 290])
    t_api.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_api)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"ReportLab successfully built PDF at {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size} bytes)")

if __name__ == '__main__':
    generate_pdf()
