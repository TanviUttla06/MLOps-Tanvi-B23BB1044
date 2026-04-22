# MLOps-Tanvi-B23BB1044

\# NLP Translation Task (Q1)



\## Model Used

Helsinki-NLP/opus-mt-bn-en



\## Steps to Run



\### Build Docker Image

docker build -t translator .



\### Run Translation

docker run --rm -v C:\\Users\\tanvi\\OneDrive\\Desktop\\Q1:/app translator



\### Evaluate BLEU Score

python evaluate.py



\## First Sentence Output

I am a student.



\## BLEU Score

0.49



\## Files Included

\- Dockerfile

\- translate.py

\- evaluate.py

\- input.txt

\- reference.txt

\- output.txt

