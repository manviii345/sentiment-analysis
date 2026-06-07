import tkinter as tk
from tkinter import font as tkfont
import cv2
from PIL import Image, ImageTk
import threading
import time

from visual_pipeline import VisualPipeline
from text_pipeline import TextPipeline
from fusion_engine import FusionEngine

BG           = '#0d1117'
CARD_BG      = '#161b22'
BORDER       = '#30363d'
TXT          = '#c9d1d9'
TXT2         = '#8b949e'
ACCENT       = '#58a6ff'
GREEN        = '#3fb950'
RED          = '#f85149'
YELLOW       = '#d29922'
ORANGE       = '#e3b341'
PURPLE       = '#bc8cff'

EMOTION_CLR = {
    'happy':    '#3fb950',
    'surprise': '#39d2c0',
    'neutral':  '#8b949e',
    'fear':     '#bc8cff',
    'disgust':  '#d29922',
    'sad':      '#58a6ff',
    'angry':    '#f85149',
}

class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Multimodal Sentiment Analysis")
        self.root.configure(bg=BG)
        self.root.geometry("1100x700")
        self.root.resizable(False, False)

        self.font_title  = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self.font_heading = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.font_body   = tkfont.Font(family="Segoe UI", size=10)
        self.font_small  = tkfont.Font(family="Segoe UI", size=9)
        self.font_big    = tkfont.Font(family="Segoe UI", size=14, weight="bold")

        self.running = True
        self.camera_ready = False
        self.models_loaded = False
        self.analysis_lock = threading.Lock()

        self.visual = VisualPipeline()
        self.text_pipe = TextPipeline()
        self.fusion = FusionEngine()

        self._build_gui()

        self.status_var.set("Loading models... please wait")

        self.init_thread = threading.Thread(target=self._initialize_models, daemon=True)
        self.init_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_gui(self):
        title_frame = tk.Frame(self.root, bg=CARD_BG, pady=8)
        title_frame.pack(fill='x', padx=0, pady=(0, 4))
        tk.Label(
            title_frame, text="Real-Time Multimodal Sentiment Analysis",
            font=self.font_title, fg=ACCENT, bg=CARD_BG
        ).pack(side='left', padx=16)
        tk.Label(
            title_frame, text="CNN + Transformer | Dynamic Fusion",
            font=self.font_small, fg=TXT2, bg=CARD_BG
        ).pack(side='right', padx=16)

        content = tk.Frame(self.root, bg=BG)
        content.pack(fill='both', expand=True, padx=10, pady=4)

        left = tk.Frame(content, bg=BG, width=500)
        left.pack(side='left', fill='both', padx=(0, 5))
        left.pack_propagate(False)

        right = tk.Frame(content, bg=BG, width=580)
        right.pack(side='right', fill='both', expand=True, padx=(5, 0))

        cam_card = self._card(left, "Live Webcam Feed")
        self.webcam_label = tk.Label(cam_card, bg='#000000', width=480, height=300)
        self.webcam_label.pack(padx=5, pady=5)

        emo_card = self._card(left, "Facial Emotion Distribution (CNN)")
        self.emotion_canvas = tk.Canvas(emo_card, bg=CARD_BG, height=145, highlightthickness=0)
        self.emotion_canvas.pack(fill='x', padx=8, pady=4)

        txt_card = self._card(right, "Text Input (Transformer)")
        self.text_input = tk.Text(
            txt_card, height=3, font=self.font_body,
            bg='#0d1117', fg=TXT, insertbackground=TXT,
            relief='solid', bd=1, wrap='word',
            highlightbackground=BORDER, highlightcolor=ACCENT
        )
        self.text_input.pack(fill='x', padx=8, pady=(4, 4))
        self.text_input.insert('1.0', 'Type a sentence here and click Analyze...')
        self.text_input.bind('<FocusIn>', self._clear_placeholder)

        self.analyze_btn = tk.Button(
            txt_card, text="Analyze Sentiment", font=self.font_heading,
            bg=ACCENT, fg='#ffffff', activebackground='#1f6feb',
            activeforeground='#ffffff', relief='flat', cursor='hand2',
            command=self._on_analyze_click, padx=16, pady=6
        )
        self.analyze_btn.pack(pady=(0, 6))

        ts_card = self._card(right, "Text Sentiment Scores")
        self.text_canvas = tk.Canvas(ts_card, bg=CARD_BG, height=70, highlightthickness=0)
        self.text_canvas.pack(fill='x', padx=8, pady=4)

        fu_card = self._card(right, "Context-Aware Dynamic Fusion")
        self.fusion_frame = tk.Frame(fu_card, bg=CARD_BG)
        self.fusion_frame.pack(fill='x', padx=8, pady=4)

        wt_row = tk.Frame(self.fusion_frame, bg=CARD_BG)
        wt_row.pack(fill='x')
        tk.Label(wt_row, text="Dynamic Weights:", font=self.font_small, fg=TXT2, bg=CARD_BG).pack(side='left')
        self.weight_label = tk.Label(wt_row, text="Visual: --% | Text: --%", font=self.font_body, fg=TXT, bg=CARD_BG)
        self.weight_label.pack(side='right')

        self.conflict_label = tk.Label(
            self.fusion_frame, text="Waiting for analysis...",
            font=self.font_body, fg=TXT2, bg=CARD_BG, wraplength=500, justify='left'
        )
        self.conflict_label.pack(fill='x', pady=(4, 0))

        final_card = self._card(right, "Final Multimodal Sentiment")
        self.final_label = tk.Label(
            final_card, text="Waiting for input...",
            font=self.font_big, fg=TXT2, bg=CARD_BG, pady=8
        )
        self.final_label.pack(fill='x', padx=8, pady=4)

        self.score_detail = tk.Label(
            final_card, text="", font=self.font_small, fg=TXT2, bg=CARD_BG
        )
        self.score_detail.pack(fill='x', padx=8, pady=(0, 6))

        status_frame = tk.Frame(self.root, bg=CARD_BG, pady=4)
        status_frame.pack(fill='x', side='bottom')
        self.status_var = tk.StringVar(value="Initializing...")
        tk.Label(
            status_frame, textvariable=self.status_var,
            font=self.font_small, fg=TXT2, bg=CARD_BG
        ).pack(side='left', padx=12)

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=BORDER, pady=1, padx=1)
        outer.pack(fill='x', pady=3)
        inner = tk.Frame(outer, bg=CARD_BG)
        inner.pack(fill='both', expand=True)
        tk.Label(
            inner, text=title, font=self.font_heading,
            fg=ACCENT, bg=CARD_BG, anchor='w'
        ).pack(fill='x', padx=8, pady=(6, 2))
        return inner

    def _clear_placeholder(self, event):
        content = self.text_input.get('1.0', 'end-1c')
        if content == 'Type a sentence here and click Analyze...':
            self.text_input.delete('1.0', tk.END)

    def _initialize_models(self):
        self.status_var.set("Starting webcam...")
        self.camera_ready = self.visual.start_camera()
        if self.camera_ready:
            self.root.after(0, self._update_webcam)
            self.status_var.set("Webcam ready. Loading RoBERTa model...")
        else:
            self.status_var.set("Webcam not found - text-only mode active")

        try:
            self.text_pipe.load_model()
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Text model error: {e}"))
            return

        self.status_var.set("Warming up DeepFace emotion model...")
        if self.camera_ready and self.visual.current_frame is not None:
            self.visual.analyze_emotion(self.visual.current_frame)

        self.models_loaded = True
        self.root.after(0, lambda: self.status_var.set("All models loaded - ready to analyze!"))

        if self.camera_ready:
            emo_thread = threading.Thread(target=self._emotion_loop, daemon=True)
            emo_thread.start()

    def _update_webcam(self):
        if not self.running:
            return

        frame = self.visual.get_frame()
        if frame is not None:
            display = frame.copy()

            if self.visual.current_emotions and self.visual.face_detected:
                dom = self.visual.dominant_emotion
                score, conf = self.visual.get_sentiment_score()
                cv2.putText(
                    display, f"{dom.upper()} ({conf*100:.0f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (88, 166, 255), 2
                )

            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            display = cv2.resize(display, (480, 300))
            img = Image.fromarray(display)
            imgtk = ImageTk.PhotoImage(image=img)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)

        self.root.after(33, self._update_webcam)

    def _emotion_loop(self):
        while self.running:
            frame = self.visual.current_frame
            if frame is not None:
                self.visual.analyze_emotion(frame)
                self.root.after(0, self._draw_emotion_bars)
            time.sleep(0.8)

    def _draw_emotion_bars(self):
        c = self.emotion_canvas
        c.delete('all')

        emotions = self.visual.current_emotions
        if emotions is None:
            c.create_text(
                240, 70, text="No face detected", fill=TXT2,
                font=self.font_body
            )
            return

        bar_w = 200
        bar_h = 16
        x_start = 110
        y = 8

        sorted_emo = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        for emo, prob in sorted_emo:
            color = EMOTION_CLR.get(emo, TXT2)
            label = f"{emo.capitalize()}"
            pct = f"{prob*100:.1f}%"

            c.create_text(x_start - 8, y + bar_h // 2, text=label,
                          anchor='e', fill=TXT, font=self.font_small)

            c.create_rectangle(x_start, y, x_start + bar_w, y + bar_h,
                               fill=BORDER, outline='')
            fill_w = max(1, int(bar_w * prob))
            c.create_rectangle(x_start, y, x_start + fill_w, y + bar_h,
                               fill=color, outline='')

            c.create_text(x_start + bar_w + 8, y + bar_h // 2, text=pct,
                          anchor='w', fill=TXT2, font=self.font_small)

            y += bar_h + 4

    def _on_analyze_click(self):
        if not self.models_loaded:
            self.status_var.set("Models still loading, please wait...")
            return

        text = self.text_input.get('1.0', 'end-1c').strip()
        if not text or text == 'Type a sentence here and click Analyze...':
            self.status_var.set("Please enter some text first.")
            return

        self.status_var.set("Analyzing...")
        self.analyze_btn.configure(state='disabled')

        threading.Thread(target=self._run_analysis, args=(text,), daemon=True).start()

    def _run_analysis(self, text):
        text_scores = self.text_pipe.analyze(text)

        self.root.after(0, lambda: self._draw_text_bars(text_scores))
        self.root.after(0, self._run_fusion)
        self.root.after(0, lambda: self.analyze_btn.configure(state='normal'))

    def _draw_text_bars(self, scores):
        c = self.text_canvas
        c.delete('all')

        bar_w = 250
        bar_h = 16
        x_start = 90
        y = 8

        colors = {'positive': GREEN, 'neutral': YELLOW, 'negative': RED}

        for label in ['positive', 'neutral', 'negative']:
            prob = scores.get(label, 0.0)
            color = colors[label]
            pct = f"{prob*100:.1f}%"

            c.create_text(x_start - 8, y + bar_h // 2, text=label.capitalize(),
                          anchor='e', fill=TXT, font=self.font_small)
            c.create_rectangle(x_start, y, x_start + bar_w, y + bar_h,
                               fill=BORDER, outline='')
            fill_w = max(1, int(bar_w * prob))
            c.create_rectangle(x_start, y, x_start + fill_w, y + bar_h,
                               fill=color, outline='')
            c.create_text(x_start + bar_w + 8, y + bar_h // 2, text=pct,
                          anchor='w', fill=TXT2, font=self.font_small)
            y += bar_h + 4

    def _run_fusion(self):
        visual_score, visual_conf = self.visual.get_sentiment_score()
        text_score, text_conf = self.text_pipe.get_sentiment_score()

        final_score, final_conf, wv, wt = self.fusion.dynamic_fusion(
            visual_score, visual_conf, text_score, text_conf
        )
        conflict, message = self.fusion.detect_conflict(visual_score, text_score)

        self.weight_label.configure(
            text=f"Visual: {wv*100:.0f}%  |  Text: {wt*100:.0f}%"
        )

        if conflict:
            self.conflict_label.configure(text=message, fg=ORANGE)
        else:
            self.conflict_label.configure(text=message, fg=GREEN)

        label = self.fusion.classify_sentiment(final_score)
        color = self.fusion.score_to_color_hex(final_score)
        
        self.final_label.configure(
            text=f"{label}",
            fg=color
        )
        self.score_detail.configure(
            text=(
                f"Score: {final_score:+.3f}  |  "
                f"Confidence: {final_conf*100:.0f}%  |  "
                f"Visual: {visual_score:+.3f}  |  "
                f"Text: {text_score:+.3f}"
            )
        )

        self.status_var.set(
            f"Analysis complete - Final: {label} ({final_score:+.3f})"
        )

    def _on_close(self):
        self.running = False
        self.visual.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
