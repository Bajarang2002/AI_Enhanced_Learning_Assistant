"""
=============================================================================
AI-Enhanced Learning Assistant - Dual Answer Sheet Evaluation Engine
=============================================================================
Compares:
  1. Correct / Reference Answer Sheet
  2. Student Answer Sheet
Computes semantic similarity, key concept coverage, completeness score,
and generates detailed diagnostic feedback with numerical scoring.
=============================================================================
"""

import os
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_file_or_content(file_obj=None, file_path=None, raw_text=None):
    """
    Extracts plain text from an uploaded file object, path, or raw text input.
    Supports .pdf and .txt.
    """
    if raw_text and raw_text.strip():
        return raw_text.strip()

    if file_path and os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            try:
                import fitz
                text = ""
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text() + "\n"
                return text.strip()
            except Exception:
                pass
        
        # Read text file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        except Exception:
            return ""

    if file_obj:
        filename = getattr(file_obj, 'filename', '')
        ext = os.path.splitext(filename)[1].lower()
        content = file_obj.read()
        file_obj.seek(0)  # Reset pointer

        if ext == '.pdf':
            try:
                import fitz
                doc = fitz.open(stream=content, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                return text.strip()
            except Exception:
                return content.decode('utf-8', errors='ignore').strip()
        else:
            try:
                import chardet
                det = chardet.detect(content)
                enc = det.get('encoding') or 'utf-8'
                return content.decode(enc, errors='replace').strip()
            except Exception:
                return content.decode('utf-8', errors='ignore').strip()

    return ""


def extract_key_concepts(text, top_n=12):
    """
    Extracts significant noun phrases and keywords from text.
    """
    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        
        # Stopwords filter
        stopwords = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'in', 'of',
            'to', 'for', 'with', 'as', 'by', 'that', 'this', 'it', 'from', 'be',
            'are', 'was', 'were', 'has', 'have', 'had', 'been', 'will', 'would',
            'can', 'could', 'should', 'shall', 'may', 'might', 'must', 'do', 'does'
        }

        # Noun chunks
        grammar = r"NP: {<JJ>*<NN|NNP|NNS>+}"
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(tagged)

        concepts = []
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                phrase = " ".join([word.lower() for word, tag in subtree.leaves()])
                if phrase not in stopwords and len(phrase) > 2 and phrase not in concepts:
                    concepts.append(phrase)

        # Fallback to single important nouns
        if len(concepts) < 5:
            for word, tag in tagged:
                w_lower = word.lower()
                if tag.startswith('NN') and w_lower not in stopwords and len(w_lower) > 2:
                    if w_lower not in concepts:
                        concepts.append(w_lower)

        return concepts[:top_n]
    except Exception:
        # Fallback regex extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in {'the', 'and', 'for', 'with', 'this', 'that', 'from'}]
        return list(dict.fromkeys(filtered))[:top_n]


def evaluate_answers(reference_answer, student_answer):
    """
    Compares reference answer sheet against student answer sheet.
    
    Returns:
        dict: Containing final score, similarity percentage, key concepts match,
              missing concepts, and detailed feedback.
    """
    if not reference_answer or not reference_answer.strip():
        raise ValueError("Reference answer sheet is empty.")
    if not student_answer or not student_answer.strip():
        raise ValueError("Student answer sheet is empty.")

    ref_clean = reference_answer.strip()
    student_clean = student_answer.strip()

    # 1. Semantic Similarity via TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([ref_clean, student_clean])
        cosine_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception:
        cosine_sim = 0.5  # Fallback baseline

    similarity_pct = round(cosine_sim * 100.0, 2)

    # 2. Key Concept Coverage
    ref_concepts = extract_key_concepts(ref_clean, top_n=10)
    student_lower = student_clean.lower()

    matched_concepts = []
    missing_concepts = []

    for concept in ref_concepts:
        if concept in student_lower:
            matched_concepts.append(concept)
        else:
            missing_concepts.append(concept)

    if ref_concepts:
        coverage_pct = round((len(matched_concepts) / len(ref_concepts)) * 100.0, 2)
    else:
        coverage_pct = 75.0

    # 3. Completeness & Length Ratio
    ref_words = len(ref_clean.split())
    student_words = len(student_clean.split())

    length_ratio = min(1.0, student_words / max(1, ref_words * 0.8))
    completeness_score = round(length_ratio * 10.0, 2)

    # 4. Final Weighted Score Calculation (out of 10)
    # 55% Semantic Similarity + 35% Keyword Coverage + 10% Length Completeness
    score_out_of_10 = round(
        (cosine_sim * 5.5) + ((coverage_pct / 100.0) * 3.5) + (length_ratio * 1.0),
        2
    )
    score_out_of_10 = max(0.5, min(10.0, score_out_of_10))
    score_out_of_100 = round(score_out_of_10 * 10.0, 1)

    # 5. Diagnostic Feedback Generation
    if score_out_of_10 >= 9.0:
        grade = "Outstanding (Grade A+)"
        feedback_summary = "Excellent answer! Demonstrates comprehensive domain understanding, accurate terminology, and complete conceptual alignment with the reference sheet."
    elif score_out_of_10 >= 7.5:
        grade = "Very Good (Grade A)"
        feedback_summary = "Well-articulated answer covering the primary concepts effectively. Minor elaboration on key terminology would achieve full marks."
    elif score_out_of_10 >= 6.0:
        grade = "Satisfactory (Grade B)"
        feedback_summary = "Good basic attempt. The answer captures the core idea but lacks depth and misses several essential technical concepts."
    elif score_out_of_10 >= 4.0:
        grade = "Needs Improvement (Grade C)"
        feedback_summary = "Partially relevant answer. Significant conceptual gaps exist compared to the reference answer sheet."
    else:
        grade = "Unsatisfactory (Grade D/F)"
        feedback_summary = "The provided answer shows very low correlation with the reference answer key. Please review the material thoroughly."

    recommendations = []
    if missing_concepts:
        top_missing = missing_concepts[:5]
        recommendations.append(f"Incorporate missing core concepts: {', '.join(top_missing)}.")
    if student_words < ref_words * 0.5:
        recommendations.append(f"Expand the explanation with more detailed definitions and examples (current length: {student_words} words vs expected: ~{ref_words} words).")
    if not recommendations:
        recommendations.append("Continue maintaining this high standard of technical writing and clarity.")

    return {
        'score': score_out_of_10,
        'score_100': score_out_of_100,
        'grade': grade,
        'similarity_pct': similarity_pct,
        'coverage_pct': coverage_pct,
        'completeness_score': completeness_score,
        'matched_concepts': matched_concepts,
        'missing_concepts': missing_concepts,
        'feedback_summary': feedback_summary,
        'recommendations': recommendations,
        'ref_word_count': ref_words,
        'student_word_count': student_words
    }
