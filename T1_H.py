import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
from collections import Counter
from ttkthemes import ThemedStyle

class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def euclidean_distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for test_vec in X_test:
            distances = [(self.euclidean_distance(test_vec, train_vec), label) for train_vec, label in zip(self.X_train, self.y_train)]
            distances.sort()
            k_nearest_labels = [label for _, label in distances[:self.k]]
            most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return predictions

def load_tfidf_results(filename):
    doc_tfidf = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            doc_line = lines[i].strip()
            if doc_line.startswith('Document:'):
                doc_id = int(doc_line.split(':')[1].strip())
                tfidf_line = lines[i+1].strip()
                tfidf_vector = [float(x) for x in tfidf_line.split(':')[1].split(',')]
                doc_tfidf[doc_id] = tfidf_vector
                i += 2
            else:
                i += 1
    return doc_tfidf

def calculate_metrics(document_labels, test_docs, predictions):
    p_labels = []
    t_labels = []
    precisions = {}
    recalls = {}
    macro_index = {}

    for i, j in zip(test_docs, predictions):
        p_labels.append(j)
        t_labels.append(document_labels[i])
    
    for t_l in set(t_labels):
        if t_l not in macro_index:
            macro_index[t_l] = {"TP":0,"FP":0,"FN":0}
        macro_index[t_l]["TP"] = sum(1 for true_label, predicted_label in zip(t_labels, p_labels) if true_label == predicted_label and true_label == t_l)
        macro_index[t_l]["FP"] = sum(1 for true_label, predicted_label in zip(t_labels, p_labels) if true_label != predicted_label and predicted_label==t_l )
        macro_index[t_l]["FN"] = sum(1 for true_label, predicted_label in zip(t_labels, p_labels) if true_label != predicted_label and  true_label == t_l)
        precisions[t_l] = macro_index[t_l]["TP"] / (macro_index[t_l]["TP"]+macro_index[t_l]["FP"]) if macro_index[t_l]["TP"]+macro_index[t_l]["FP"] != 0 else 0
        recalls[t_l] = macro_index[t_l]["TP"] / (macro_index[t_l]["TP"]+macro_index[t_l]["FN"]) if macro_index[t_l]["TP"]+macro_index[t_l]["FN"] != 0 else 0

    precision = sum(precisions[i] for i in precisions)/len(precisions)
    recall = sum(recalls[i] for i in recalls)/len(recalls)
    accuracy = sum(macro_index[label]["TP"] for label in macro_index) / len(predictions)
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return precision, recall, accuracy, f1_score

def train_and_evaluate(k_value, text_widget):
    # Load TF-IDF results
    doc_tfidf = load_tfidf_results("tfidf_results.txt")

    # Document labels
    document_labels = {
        1: "Explainable Artificial Intelligence",
        2: "Explainable Artificial Intelligence",
        3: "Explainable Artificial Intelligence",
        7: "Explainable Artificial Intelligence",
        8: "Heart Failure",
        9: "Heart Failure",
        11: "Heart Failure",
        12: "Time Series Forecasting",
        13: "Time Series Forecasting",
        14: "Time Series Forecasting",
        15: "Time Series Forecasting",
        16: "Time Series Forecasting",
        17: "Transformer Model",
        18: "Transformer Model",
        21: "Transformer Model",
        22: "Feature Selection",
        23: "Feature Selection",
        24: "Feature Selection",
        25: "Feature Selection",
        26: "Feature Selection"
    }

    # Test documents hardcoded
    test_docs = [4, 5, 6, 10, 19, 20]

    # Extract vectors for train and test data
    X_train = np.array([doc_tfidf[doc] for doc in document_labels.keys()])
    X_test = np.array([doc_tfidf[doc] for doc in test_docs])

    # Extract labels for train data
    y_train = list(document_labels.values())

    # Instantiate and fit the KNN classifier
    knn_classifier = KNNClassifier(k=k_value)
    knn_classifier.fit(X_train, y_train)

    # Make predictions
    predictions = knn_classifier.predict(X_test)

    # Calculate metrics
    precision, recall, accuracy, f1_score = calculate_metrics(document_labels, test_docs, predictions)

    # Display scores
    evaluation_results = f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nAccuracy: {accuracy:.4f}\nF1 Score: {f1_score:.4f}"
    text_widget.config(state=tk.NORMAL)
    text_widget.delete('1.0', tk.END)
    text_widget.insert(tk.END, evaluation_results)

    # Display test documents with labels
    test_document_labels = [f"Document {doc}: {document_labels[doc]}\n" for doc in test_docs]
    test_documents_text = "".join(test_document_labels)
    text_widget.insert(tk.END, "\n\nTest Documents with Labels:\n")
    text_widget.insert(tk.END, test_documents_text)
    text_widget.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    root.title("KNN Classifier Evaluation")
    style = ThemedStyle(root)
    style.set_theme("plastik")

    # Create a frame for input
    input_frame = ttk.Frame(root, padding="20")
    input_frame.pack()

    # Label and Entry for k value
    ttk.Label(input_frame, text="Enter k value:").grid(row=0, column=0, padx=5, pady=5)
    k_entry = ttk.Entry(input_frame)
    k_entry.grid(row=0, column=1, padx=5, pady=5)

    # Button to start evaluation
    evaluate_button = ttk.Button(input_frame, text="Evaluate")
    evaluate_button.grid(row=1, column=0, columnspan=2, pady=10)
    evaluate_button.config(command=lambda: train_and_evaluate(int(k_entry.get()), text_widget))

    # Text widget to display evaluation results and test documents
    text_widget = tk.Text(root, width=70, height=20)
    text_widget.pack(expand=True, fill=tk.BOTH)
    text_widget.config(state=tk.DISABLED)

    root.mainloop()

if __name__ == "__main__":
    main()
