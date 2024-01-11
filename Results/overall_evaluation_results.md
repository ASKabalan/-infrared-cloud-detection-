|             | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 Score | Error Rate (ER) | BinaryCrossEntropy Loss | IOU       | Mean AUC   |
|-------------|---------------|----------------|-------------|---------------|-----------------|-------------------------|-----------|------------|
| **LWIRISEG**| **0.9464**    | **0.9593**     | **0.9797**  | **0.9694**    | **0.0536**      | **0.1292**              | **0.9406**| **0.9763** |
| HYTA        | 0.4585        | 0.4317         | 0.9607      | 0.5957        | 0.5415          | 8.1540                  | 0.4242    | 0.5313     |
| HYTA FT     | 0.7753        | 0.8066         | 0.6444      | 0.7164        | 0.2247          | 0.5010                  | 0.5582    | 0.7614     |
| HYTA FS     | 0.5599        | 1.0000         | 0.0012      | 0.0025        | 0.4401          | 0.6376                  | 0.0012    | 0.5006     |
| SWIMSEG     | 0.5751        | 0.5633         | 0.9883      | 0.7176        | 0.4249          | 6.4259                  | 0.5595    | 0.5332     |
| SWIMSEG FT  | 0.8456        | 0.8500         | 0.8709      | 0.8603        | 0.1544          | 0.3425                  | 0.7549    | 0.8430     |
| SWIMSEG FS  | 0.8865        | 0.8794         | 0.9185      | 0.8985        | 0.1135          | 0.2680                  | 0.8157    | 0.9576     |
| SWINSEG     | 0.6706        | 0.5827         | 0.9639      | 0.7263        | 0.3294          | 4.7272                  | 0.5703    | 0.6955     |
| SWINSEG FT  | 0.9164        | 0.9363         | 0.8753      | 0.9047        | 0.0836          | 0.1999                  | 0.8261    | 0.9129     |
| SWINSEG FS  | 0.9325        | 0.8843         | 0.9793      | 0.9293        | 0.0675          | 0.1670                  | 0.8680    | 0.9364     |




```cpp
\begin{table*}[t]
    \begin{center}
        \caption{Evaluation metrics for the proposed segmentation model on publicly available state-of-the-art datasets. Note that RGB color images are transformed into gray-scale images as the IRIS-CloudDeep segmentation model is optimized for this type of data. Best values are denoted in bold font. (A = accuracy, P = precision, R = Recall, F1 = F1-score, ER = error rate, BC Loss = binary cross-entropy loss, IoU = intersection over union, AUC = area under the curve).}
        \begin{tabular}{c c c c c c c c c} 
        \tophline \hline

         Dataset & A [\%] & P [\%] & R [\%] & F1 [\%] & ER [\%] & BC Loss & IoU & AUC \\ [1.0ex]
         \hline
         \textbf{LWIRISEG} & \textbf{94.64} & \textbf{95.93} & \textbf{97.97} & \textbf{96.94} & \textbf{5.36} & \textbf{0.1292} & \textbf{94.06} & \textbf{97.63} \\ [1.0ex]
         HYTA & 45.85 & 43.17 & 96.07 & 59.57 & 54.15 & 8.1540 & 42.42 & 53.13 \\ [1.0ex]
         HYTA FT & 77.53 & 80.66 & 64.44 & 71.64 & 22.47 & 0.5010 & 55.82 & 76.14 \\ [1.0ex]
         HYTA FS & 55.99 & 100.00 & 0.12 & 0.25 & 44.01 & 0.6376 & 0.12 & 50.06 \\ [1.0ex]
         SWIMSEG & 57.51 & 56.33 & 98.83 & 71.76 & 42.49 & 6.4259 & 55.95 & 53.32 \\ [1.0ex]
         SWIMSEG FT & 84.56 & 85.00 & 87.09 & 86.03 & 15.44 & 0.3425 & 75.49 & 84.30 \\ [1.0ex]
         SWIMSEG FS & 88.65 & 87.94 & 91.85 & 89.85 & 11.35 & 0.2680 & 81.57 & 95.76 \\ [1.0ex]
         SWINSEG & 67.06 & 58.27 & 96.39 & 72.63 & 32.94 & 4.7272 & 57.03 & 69.55 \\ [1.0ex]
         SWINSEG FT & 91.64 & 93.63 & 87.53 & 90.47 & 8.36 & 0.1999 & 82.61 & 91.29 \\ [1.0ex]
         SWINSEG FS & 93.25 & 88.43 & 97.93 & 92.93 & 6.75 & 0.1670 & 86.80 & 93.64 \\ [1.0ex]
         \hline
        \end{tabular}
        \label{tab:datasets_comparison}
        \belowtable{} % You can add footnotes or explanations here if needed
    \end{center}
\end{table*}
```
