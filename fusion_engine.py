class FusionEngine:
    CONFLICT_THRESHOLD = 0.15

    def __init__(self):
        self.conflict_detected = False
        self.conflict_message = ""
        self.last_result = None

    def dynamic_fusion(self, visual_score, visual_conf, text_score, text_conf):
        total_conf = visual_conf + text_conf

        if total_conf < 0.01:
            return 0.0, 0.0, 0.5, 0.5

        wv = visual_conf / total_conf
        wt = text_conf / total_conf

        final_score = wv * visual_score + wt * text_score
        final_conf = wv * visual_conf + wt * text_conf
        final_score = max(-1.0, min(1.0, final_score))

        self.last_result = {
            'final_score': round(final_score, 4),
            'final_confidence': round(final_conf, 4),
            'weight_visual': round(wv, 4),
            'weight_text': round(wt, 4),
            'visual_score': round(visual_score, 4),
            'text_score': round(text_score, 4),
        }

        return final_score, final_conf, wv, wt

    def detect_conflict(self, visual_score, text_score):
        t = self.CONFLICT_THRESHOLD

        positive_face_negative_text = (visual_score > t and text_score < -t)
        negative_face_positive_text = (visual_score < -t and text_score > t)

        if positive_face_negative_text:
            self.conflict_detected = True
            self.conflict_message = "Conflict: Face Positive, Text Negative"
        elif negative_face_positive_text:
            self.conflict_detected = True
            self.conflict_message = "Conflict: Face Negative, Text Positive"
        else:
            self.conflict_detected = False
            self.conflict_message = "No conflict"

        return self.conflict_detected, self.conflict_message

    @staticmethod
    def classify_sentiment(score):
        if score > 0.35:
            return "Positive"
        elif score > 0.12:
            return "Slightly Positive"
        elif score > -0.12:
            return "Neutral"
        elif score > -0.35:
            return "Slightly Negative"
        else:
            return "Negative"

    @staticmethod
    def score_to_color_hex(score):
        t = (score + 1.0) / 2.0
        t = max(0.0, min(1.0, t))

        if t < 0.5:
            r = 248
            g = int(81 + (210 - 81) * (t / 0.5))
            b = int(73 + (34 - 73) * (t / 0.5))
        else:
            r = int(210 + (63 - 210) * ((t - 0.5) / 0.5))
            g = int(210 + (185 - 210) * ((t - 0.5) / 0.5))
            b = int(34 + (80 - 34) * ((t - 0.5) / 0.5))

        return f"#{r:02x}{g:02x}{b:02x}"
