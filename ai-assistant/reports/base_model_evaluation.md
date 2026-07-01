# Base Model Evaluation — Before Fine-Tuning

**Model:** TinyLlama-1.1B (unsloth/tinyllama-bnb-4bit)  
**Stage:** Pre-training evaluation (zero-shot)  
**Date:** 2026-06-30  
**Domain:** Healthcare FAQ Assistant

---

## Evaluation Summary

The base TinyLlama-1.1B model was tested on 10 domain-specific healthcare questions without any fine-tuning. The responses were captured as-is to establish a baseline for comparison after instruction fine-tuning and DPO alignment.

---

## Evaluation Table

| # | Question | Base Model Answer | Problem Identified |
|---|----------|-------------------|-------------------|
| 1 | What are the symptoms of type 2 diabetes? | "Diabetes is a disease that affects the blood. You should see a doctor if you feel unwell. There are many types of diabetes and it can affect anyone." | Generic and vague. No specific symptoms listed. No mention of increased thirst, frequent urination, fatigue, or blurred vision. Not medically useful. |
| 2 | How can I lower my blood pressure naturally? | "Blood pressure is important for health. You can try eating less salt and exercising more. It is also important to relax and not get too stressed. Talk to a doctor for more advice." | Extremely generic. Missing DASH diet, specific exercise guidance, weight management, alcohol limitation, and sleep. Not actionable. |
| 3 | What should I do during a heart attack? | "A heart attack is very serious. You should call a doctor or go to the hospital. Try to stay calm and breathe slowly. Lie down and wait for help." | Critically incomplete. No mention of calling 911 immediately. No mention of aspirin. Dangerous advice to 'breathe slowly' and wait — delays are fatal in cardiac emergencies. |
| 4 | How much exercise should adults get per week? | "Adults should exercise regularly to stay healthy. Walking, running, and swimming are all good forms of exercise. Exercise helps your heart and keeps your weight normal." | No specific guidelines provided. No mention of the 150 minutes/week moderate or 75 minutes/week vigorous recommendation. Not evidence-based. |
| 5 | What is the difference between a cold and the flu? | "A cold and the flu are both illnesses that affect the respiratory system. They have similar symptoms. Both usually get better on their own with rest and fluids." | Fails to differentiate meaningfully. No mention of severity differences, fever patterns, body aches, flu vaccine, or antivirals. Medically inaccurate in not distinguishing severity. |
| 6 | What foods should I eat for a healthy heart? | "Eating healthy food is good for your heart. Fruits and vegetables are important. You should avoid junk food and eat balanced meals. Drink plenty of water." | Too vague to be useful. No mention of omega-3s, whole grains, specific healthy fats, sodium limits, or the DASH diet. Does not guide the reader toward specific food choices. |
| 7 | How can I improve my sleep quality? | "Sleep is important for health. You should try to sleep at a regular time every night. Avoid caffeine before bed. Make sure your bedroom is comfortable and dark." | Partially correct but superficial. Misses screen avoidance, alcohol effects, sleep hygiene details, and when to seek professional help. Generic enough to apply to anyone and is not specifically helpful. |
| 8 | What are the warning signs of a stroke? | "A stroke is a medical emergency that affects the brain. Symptoms include dizziness and confusion. You should call a doctor if you think someone is having a stroke." | Dangerously incomplete. No mention of the FAST acronym (Face drooping, Arm weakness, Speech difficulty, Time to call 911). Dizziness alone as a symptom is insufficient. 'Call a doctor' is inadequate — 911 must be called immediately. |
| 9 | How can I prevent kidney stones? | "Kidney stones are painful. Drinking lots of water can help prevent them. You should avoid certain foods if you are prone to kidney stones. Speak to your doctor for more details." | No specific dietary guidance (sodium limits, oxalate restriction, calcium recommendations). No mention of stone type differences. Generic advice with no actionable detail. |
| 10 | What is the recommended daily water intake? | "You should drink plenty of water every day. The exact amount depends on your body size and activity level. A good rule of thumb is to drink when you feel thirsty." | Does not state the commonly cited guideline (approximately 8 glasses / 2 liters per day). Drinking 'when thirsty' is not always reliable, especially for older adults and athletes. No mention of urine color as an indicator. |

---

## Key Observations

1. **Generic responses:** The base model consistently gives vague, non-specific answers that lack medical accuracy or actionable guidance.
2. **Missing domain specificity:** Healthcare terminology, clinical guidelines, and evidence-based recommendations are absent.
3. **Safety risks:** For critical questions (heart attack, stroke), the base model provides dangerously incomplete advice — for example, not specifying "call 911 immediately."
4. **No structure:** Responses lack organized formatting, bullet points, or numbered steps that aid healthcare communication.
5. **No quantitative guidance:** The model fails to provide specific values (e.g., 150 min/week exercise, 2,300 mg sodium limit, 7–9 hours sleep).
6. **Over-reliance on 'talk to a doctor':** While appropriate, this is used as a catch-all to avoid specific answers rather than complementing them.

---

## Conclusion

The base TinyLlama-1.1B model is not suitable for healthcare FAQ use without fine-tuning. Its responses are generic, lack domain accuracy, and in some cases could be harmful if relied upon in real healthcare contexts. Instruction fine-tuning on the curated healthcare Q&A dataset is necessary to improve specificity, accuracy, and clinical usefulness.
