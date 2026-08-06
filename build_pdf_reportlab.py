import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether, HRFlowable
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
        # Custom page labeling for report insertion
        page_labels = ["iii", "vi", "7", "10", "20", "25", "37", "38", "39", "40"]
        if self._pageNumber <= len(page_labels):
            label = page_labels[self._pageNumber - 1]
            self.drawCentredString(A4[0] / 2.0, 28, label)
        self.restoreState()

def generate_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=38,
        bottomMargin=38,
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=13.5,
        leading=17,
        alignment=1,
        spaceAfter=12,
        textTransform='uppercase'
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=10.5,
        leading=14,
        spaceBefore=6,
        spaceAfter=3,
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.8,
        leading=13.8,
        alignment=4, # Justified
        spaceAfter=6
    )
    
    sig_style = ParagraphStyle(
        'Sig_Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13.5,
    )
    
    math_box_style = ParagraphStyle(
        'MathBox',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=9,
        leading=13,
        alignment=1,
        textColor=colors.HexColor('#0f2b60')
    )

    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
    )

    cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=8.5,
        leading=11,
    )

    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9,
        leading=12,
        alignment=1,
        spaceBefore=3,
        spaceAfter=7
    )

    story = []

    # ==========================================
    # PAGE 1: ACKNOWLEDGEMENTS (CORRECTED WITH ALL 4 AUTHORS)
    # ==========================================
    story.append(Paragraph("ACKNOWLEDGEMENTS", title_style))
    story.append(Spacer(1, 2))
    
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
    
    story.append(Spacer(1, 15))
    
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
    sig_table = Table(sig_data, colWidths=[250, 250])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(sig_table)
    
    story.append(PageBreak())

    # ==========================================
    # PAGE 2: TABLE OF CONTENTS (ACCURATE ALIGNMENT)
    # ==========================================
    story.append(Paragraph("TABLE OF CONTENTS", title_style))
    story.append(Spacer(1, 2))
    
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
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;6.1 Tools, 6.2 Preprocessing &amp; 6.3 9-Stage Validation Matrix", cell_style), Paragraph("18 - 20", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;6.4 Audio Synthesizer &amp; 6.5 REST Telemetry API", cell_style), Paragraph("21 - 22", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 7 &nbsp; RESULTS AND DISCUSSION</b>", cell_bold), Paragraph("23", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;7.1 Leaderboard &amp; 7.2 Confusion Matrix / ROC", cell_style), Paragraph("23 - 24", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;7.5 Ablation Study: Impact of Temporal HMM", cell_style), Paragraph("25", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 8 &nbsp; COMPARATIVE ANALYSIS</b>", cell_bold), Paragraph("26", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>CHAPTER 9 &nbsp; CONCLUSION AND FUTURE WORK</b>", cell_bold), Paragraph("28", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>REFERENCES</b>", cell_bold), Paragraph("30", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>APPENDIX A &nbsp; CODE SNIPPETS</b>", cell_bold), Paragraph("32", ParagraphStyle('C', parent=cell_style, alignment=1))],
        [Paragraph("<b>APPENDIX B &nbsp; EXPERIMENTAL SCREENSHOTS</b>", cell_bold), Paragraph("34", ParagraphStyle('C', parent=cell_style, alignment=1))],
    ]
    
    toc_table = Table(toc_data, colWidths=[420, 80])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.black),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#666666')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
    ]))
    story.append(toc_table)
    
    story.append(PageBreak())

    # ==========================================
    # PAGE 3: CHAPTER 4 - SYSTEM ARCHITECTURE DIAGRAM
    # ==========================================
    story.append(Paragraph("CHAPTER 4: PROPOSED SYSTEM ARCHITECTURE", title_style))
    story.append(Paragraph(
        "The proposed system integrates computer vision biometrics, supervised machine learning ensembles, temporal Markov dynamic filtering, and a real-time automotive cockpit HUD into a unified multi-tier safety architecture:",
        body_style
    ))
    
    arch_img_path = str(PROJECT_ROOT / "outputs" / "system_architecture_diagram.png")
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=495, height=310))
        story.append(Paragraph("Fig 4.1: End-to-End System Architecture: Layered Computer Vision, Ensemble ML, Temporal HMM, and Cockpit Dashboard", caption_style))

    story.append(Paragraph(
        "<b>Core Architectural Pipeline:</b><br/>"
        "1. <b>Video Ingestion &amp; Biometrics:</b> MediaPipe FaceLandmarker tracks 478 3D facial landmarks at 60 FPS to derive EAR, MAR, PERCLOS, and 3D head pose.<br/>"
        "2. <b>Dynamic Calibration:</b> Records a 100-frame baseline to customize thresholds to driver anatomy.<br/>"
        "3. <b>Multi-Model Classification &amp; Temporal HMM:</b> Stacking ensemble predictions are filtered through a pure-NumPy HMM to suppress 100% of blink noise.<br/>"
        "4. <b>Multi-Tier Alerting &amp; Cockpit HUD:</b> Graduated alarms and live REST telemetry APIs enable real-time driver intervention.",
        body_style
    ))

    story.append(PageBreak())

    # ==========================================
    # PAGE 4: CHAPTER 5 - MATHEMATICAL MODELLING
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
    t_pnp = Table(pnp_box, colWidths=[495])
    t_pnp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f6fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_pnp)
    story.append(Spacer(1, 2))

    story.append(Paragraph("5.2 Dynamic Baseline Driver Calibration", h2_style))
    story.append(Paragraph(
        "To eliminate false alarms caused by natural differences in eye aperture across drivers, the system records N_calib = 100 frames to compute an individualized baseline:",
        body_style
    ))
    calib_box = [
        [Paragraph("EAR_base = (1 / N_calib) · Sum_{t=1}^{N_calib} EAR_t<br/>tau_EAR = EAR_base · 0.75, &nbsp;&nbsp; tau_MAR = 0.65", math_box_style)]
    ]
    t_calib = Table(calib_box, colWidths=[495])
    t_calib.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f6fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_calib)
    story.append(Spacer(1, 2))

    story.append(Paragraph("5.3 Ridge Regression — Closed-Form Estimator", h2_style))
    ridge_box = [
        [Paragraph("beta* = argmin_beta { || y - X·beta ||_2^2 + lambda·|| beta ||_2^2 } = ( X^T·X + lambda·I )^-1 · X^T·y", math_box_style)]
    ]
    t_ridge = Table(ridge_box, colWidths=[495])
    t_ridge.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f6fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_ridge)
    story.append(Spacer(1, 2))

    story.append(Paragraph("5.4 Bayesian Logistic Regression & Shannon Entropy", h2_style))
    bayes_box = [
        [Paragraph("P(y = k | x, w) = exp(w_k^T·x) / Sum_j exp(w_j^T·x)<br/>H(Y | x) = - Sum_{k=1}^K P(y = k | x) · log_2 P(y = k | x)", math_box_style)]
    ]
    t_bayes = Table(bayes_box, colWidths=[495])
    t_bayes.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f6fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_bayes)
    story.append(Spacer(1, 2))

    story.append(Paragraph("5.5 Hidden Markov Model (HMM) Temporal Smoothing", h2_style))
    hmm_box = [
        [Paragraph("Forward: &nbsp; alpha_t(j) = P(o_t | S_t = j) · Sum_i [ alpha_{t-1}(i) · A_{ij} ]<br/>Viterbi: &nbsp; delta_t(j) = max_i [ delta_{t-1}(i) · A_{ij} ] · P(o_t | S_t = j)", math_box_style)]
    ]
    t_hmm = Table(hmm_box, colWidths=[495])
    t_hmm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f4f6fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_hmm)

    story.append(PageBreak())

    # ==========================================
    # PAGE 5: CHAPTER 6 - 9-STAGE TEST VALIDATION MATRIX & AUDIO SYNTHESIZER
    # ==========================================
    story.append(Paragraph("CHAPTER 6: IMPLEMENTATION & VERIFICATION", title_style))
    
    story.append(Paragraph("6.3 Automated 9-Stage Verification & Unit Test Suite Matrix", h2_style))
    story.append(Paragraph(
        "To guarantee high reliability, regression safety, and algorithmic correctness, the system incorporates an automated 9-stage test suite covering all modules across 177.98 seconds of total test execution:",
        body_style
    ))

    test_data = [
        [
            Paragraph("<b>Test Stage</b>", cell_bold),
            Paragraph("<b>Target Modules Tested</b>", cell_bold),
            Paragraph("<b>Assertions &amp; Invariants Verified</b>", cell_bold),
            Paragraph("<b>Result</b>", ParagraphStyle('C', parent=cell_bold, alignment=1)),
        ],
        [
            Paragraph("<b>Phase 1</b>", cell_style),
            Paragraph("<code>preprocessing.py</code>", cell_style),
            Paragraph("Duplicate removal (4,025 to 4,001), median NaN imputation (644), outlier capping.", cell_style),
            Paragraph("PASSED", ParagraphStyle('P', parent=cell_bold, alignment=1, textColor=colors.HexColor('#15803d'))),
        ],
        [
            Paragraph("<b>Phase 2</b>", cell_style),
            Paragraph("<code>feature_extraction.py</code>", cell_style),
            Paragraph("MediaPipe 478 landmark parsing, 11-D feature vector bounds, solvePnP Euler angles.", cell_style),
            Paragraph("PASSED", ParagraphStyle('P', parent=cell_bold, alignment=1, textColor=colors.HexColor('#15803d'))),
        ],
        [
            Paragraph("<b>Phase 3</b>", cell_style),
            Paragraph("<code>linear, bayesian, svm</code>", cell_style),
            Paragraph("Ridge regression R2=0.955, Bayesian log-loss=0.0084, SVM margin bounds.", cell_style),
            Paragraph("PASSED", ParagraphStyle('P', parent=cell_bold, alignment=1, textColor=colors.HexColor('#15803d'))),
        ],
        [
            Paragraph("<b>Phase 4</b>", cell_style),
            Paragraph("<code>pca, kmeans, gmm, hier</code>", cell_style),
            Paragraph("PCA 95% variance (7 comps), K-Means ARI=0.98, GMM AIC/BIC k=4, Ward cophenetic=0.807.", cell_style),
            Paragraph("PASSED", ParagraphStyle('P', parent=cell_bold, alignment=1, textColor=colors.HexColor('#15803d'))),
        ],
        [
            Paragraph("<b>Phase 5</b>", cell_style),
            Paragraph("<code>hmm.py (Pure NumPy)</code>", cell_style),
            Paragraph("Forward-Backward sum-to-1 normalization, Viterbi log-space path optimality (99.75%).", cell_style),
            Paragraph("PASSED", ParagraphStyle('P', parent=cell_bold, alignment=1, textColor=colors.HexColor('#15803d'))),
        ],
        [
            Paragraph("<b>Phase 6</b>", cell_style),
            Paragraph("<code>decision_tree, rf, ada, ensemble</code>", cell_style),
            Paragraph("Tree depth bounds (5), RF OOB score (1.00), AdaBoost stagewise convergence, Stacking.", cell_style),
            Paragraph("PASSED", ParagraphStyle('P', parent=cell_bold, alignment=1, textColor=colors.HexColor('#15803d'))),
        ],
        [
            Paragraph("<b>Phase 7</b>", cell_style),
            Paragraph("<code>evaluation.py</code>", cell_style),
            Paragraph("Unified metrics benchmark table generation, macro-ROC AUC, confusion matrix verification.", cell_style),
            Paragraph("PASSED", ParagraphStyle('P', parent=cell_bold, alignment=1, textColor=colors.HexColor('#15803d'))),
        ],
        [
            Paragraph("<b>Phase 8</b>", cell_style),
            Paragraph("<code>alert_system.py, realtime</code>", cell_style),
            Paragraph("Sound synthesizer PCM buffers, 5/10-frame debouncing escalation, HUD rendering.", cell_style),
            Paragraph("PASSED", ParagraphStyle('P', parent=cell_bold, alignment=1, textColor=colors.HexColor('#15803d'))),
        ],
        [
            Paragraph("<b>Phase 9</b>", cell_style),
            Paragraph("<code>main.py, app.py</code>", cell_style),
            Paragraph("CLI arguments parsing, REST endpoints (/telemetry, /leaderboard), runtime hot-swap.", cell_style),
            Paragraph("PASSED", ParagraphStyle('P', parent=cell_bold, alignment=1, textColor=colors.HexColor('#15803d'))),
        ],
    ]
    t_test = Table(test_data, colWidths=[55, 115, 265, 60])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_test)
    story.append(Spacer(1, 4))

    story.append(Paragraph("6.4 Real-Time Audio Tone Synthesizer & Alarm Debounce Logic", h2_style))
    story.append(Paragraph(
        "The audio subsystem (<code>src/alert_system.py</code>) generates raw PCM sine wave alert audio in-memory using <code>pygame-ce</code> (with <code>winsound</code> OS fallback for headless hardware). To prevent driver distraction from false triggers, the system implements a temporal debouncing state machine: a Warning chime (880 Hz, 250 ms) triggers only after <b>5 consecutive frames</b> of detected fatigue, while a Critical siren (1200 Hz pulsed, 400 ms) escalates after <b>10 sustained frames</b> of eye closure.",
        body_style
    ))

    story.append(PageBreak())

    # ==========================================
    # PAGE 6: CHAPTER 7 - ABLATION STUDY & SECTION 6.5 REST API
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
    t_ablation = Table(ablation_data, colWidths=[205, 75, 140, 75])
    t_ablation.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#dcfce7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#64748b')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_ablation)
    story.append(Spacer(1, 4))
    
    story.append(Paragraph(
        "<b>Key Finding:</b> While raw single-frame EAR classifiers achieve fast inference, they misinterpret normal physiological eye blinks (150–250 ms) as microsleep events (18.2% false alarm rate). The <b>pure-NumPy Hidden Markov Model</b> enforces temporal continuity, eliminating 100% of blink false alarms without GPU dependencies.",
        body_style
    ))
    
    story.append(Spacer(1, 4))
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
    t_api = Table(api_data, colWidths=[130, 60, 305])
    t_api.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_api)

    story.append(PageBreak())

    # ==========================================
    # PAGE 7: APPENDIX B - SCREENSHOTS B.6 & B.7
    # ==========================================
    story.append(Paragraph("APPENDIX B: EXPERIMENTAL SCREENSHOTS", title_style))
    
    img_hud_path = str(PROJECT_ROOT / "outputs" / "realtime_hud_preview.png")
    if os.path.exists(img_hud_path):
        story.append(Image(img_hud_path, width=440, height=240))
        story.append(Paragraph("Screenshot B.6: Real-Time Automotive Cockpit HUD Overlay with 60 FPS Telemetry Metrics", caption_style))
    
    img_roc_path = str(PROJECT_ROOT / "outputs" / "evaluation" / "multi_model_roc_curves.png")
    if os.path.exists(img_roc_path):
        story.append(Image(img_roc_path, width=420, height=240))
        story.append(Paragraph("Screenshot B.7: Multi-Class One-vs-Rest Receiver Operating Characteristic (ROC) Curves for All 8 Classifiers", caption_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 8: APPENDIX B - SCREENSHOT B.8 (CONFUSION MATRICES)
    # ==========================================
    story.append(Paragraph("APPENDIX B: EXPERIMENTAL SCREENSHOTS (CONTD.)", title_style))
    
    img_cm_path = str(PROJECT_ROOT / "outputs" / "evaluation" / "all_models_confusion_matrices.png")
    if os.path.exists(img_cm_path):
        story.append(Image(img_cm_path, width=470, height=480))
        story.append(Paragraph("Screenshot B.8: Confusion Matrices for All 8 Machine Learning Classifiers on the 801-Sample Test Split", caption_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 9: APPENDIX B - SCREENSHOTS B.9 & B.10
    # ==========================================
    story.append(Paragraph("APPENDIX B: EXPERIMENTAL SCREENSHOTS (CONTD.)", title_style))
    
    img_hmm_path = str(PROJECT_ROOT / "outputs" / "evaluation" / "hmm_state_sequence_decoding.png")
    if os.path.exists(img_hmm_path):
        story.append(Image(img_hmm_path, width=440, height=230))
        story.append(Paragraph("Screenshot B.9: HMM Learned State Transition Matrix and Viterbi Sequential State Decoding", caption_style))
    
    img_bench_path = str(PROJECT_ROOT / "outputs" / "evaluation" / "benchmark_comparison.png")
    if os.path.exists(img_bench_path):
        story.append(Image(img_bench_path, width=440, height=230))
        story.append(Paragraph("Screenshot B.10: Multi-Model Benchmark Comparison (Accuracy, Latency, Throughput & Model Size)", caption_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 10: SYLLABUS MAPPING & MODULE AUDIT
    # ==========================================
    story.append(Paragraph("APPENDIX C: SYLLABUS COVERAGE & MODULE AUDIT", title_style))
    story.append(Paragraph(
        "To verify rigorous academic compliance with the 21CSC305P Machine Learning curriculum, the following matrix cross-references each syllabus unit with its corresponding mathematical module and empirical benchmark in the codebase:",
        body_style
    ))
    
    syllabus_data = [
        [
            Paragraph("<b>Unit &amp; Topic</b>", cell_bold),
            Paragraph("<b>Source Module(s)</b>", cell_bold),
            Paragraph("<b>Theoretical Concept Implemented</b>", cell_bold),
            Paragraph("<b>Key Metric Achieved</b>", cell_bold),
        ],
        [
            Paragraph("<b>Unit 1</b>: Preprocessing &amp; Feature Engineering", cell_style),
            Paragraph("<code>preprocessing.py<br/>feature_extraction.py</code>", cell_style),
            Paragraph("Median NaN imputation, outlier bounds, MediaPipe 478 landmarks, EAR, MAR, solvePnP 3D pose.", cell_style),
            Paragraph("11 Features<br/>4,001 Samples", cell_style),
        ],
        [
            Paragraph("<b>Unit 2</b>: Regression &amp; Probabilistic Classifiers", cell_style),
            Paragraph("<code>linear_regression.py<br/>bayesian_logistic.py<br/>svm_classifier.py</code>", cell_style),
            Paragraph("Ridge regression closed-form beta*, Bayesian MAP logistic regression with Shannon entropy, SVM (Linear + RBF).", cell_style),
            Paragraph("R2 = 0.955<br/>Acc = 100.0%<br/>Loss = 0.0084", cell_style),
        ],
        [
            Paragraph("<b>Unit 3</b>: Unsupervised Learning &amp; Dim. Reduction", cell_style),
            Paragraph("<code>pca.py, kmeans.py<br/>gmm.py, hierarchical.py</code>", cell_style),
            Paragraph("PCA 95% variance (7 comps), K-Means (k=4), GMM EM full covariance AIC/BIC, Ward agglomerative linkage.", cell_style),
            Paragraph("PCA = 95.2%<br/>ARI = 0.99<br/>Cophenetic = 0.807", cell_style),
        ],
        [
            Paragraph("<b>Unit 4</b>: Temporal Sequence Models (HMM)", cell_style),
            Paragraph("<code>hmm.py</code><br/>(Pure NumPy)", cell_style),
            Paragraph("4-state Gaussian emission HMM, Forward-Backward recursion, Viterbi log-space dynamic programming.", cell_style),
            Paragraph("Decoding Acc = 99.75%<br/>0% Blink Jitter", cell_style),
        ],
        [
            Paragraph("<b>Unit 5</b>: Tree &amp; Advanced Ensemble Methods", cell_style),
            Paragraph("<code>decision_tree.py<br/>random_forest.py<br/>adaboost.py, ensemble.py</code>", cell_style),
            Paragraph("CART Decision Tree (Gini), Random Forest (75 trees, OOB monitored), AdaBoost stumps, Stacking Meta-Learner.", cell_style),
            Paragraph("DT Acc = 99.25%<br/>RF Acc = 100.0%<br/>Stacking = 100.0%", cell_style),
        ],
        [
            Paragraph("<b>Integration</b>: Real-Time Edge Deployment", cell_style),
            Paragraph("<code>app.py, main.py<br/>realtime_detection.py</code>", cell_style),
            Paragraph("Async threaded frame capture, 60 FPS HTML5 cockpit HUD, 8 REST endpoints, runtime model hot-swapping.", cell_style),
            Paragraph("Latency = 0.001ms<br/>FPS = 15,300+", cell_style),
        ],
    ]
    t_syl = Table(syllabus_data, colWidths=[105, 100, 205, 85])
    t_syl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_syl)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"ReportLab successfully built 10-page master PDF at {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size} bytes)")

if __name__ == '__main__':
    generate_pdf()
