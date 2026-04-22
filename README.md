# MLOps-Tanvi-B23BB1044

 NLP Translation Task (Q1)



 Model Used

Helsinki-NLP/opus-mt-bn-en



 Steps to Run



 Build Docker Image

docker build -t translator .



 Run Translation

docker run --rm -v C:\\Users\\tanvi\\OneDrive\\Desktop\\Q1:/app translator



 Evaluate BLEU Score

python evaluate.py



First Sentence Output

I have a test today.



 BLEU Score

0.48946657165068425


 Files Included

\- Dockerfile

\- translate.py

\- evaluate.py

\- input.txt

\- reference.txt

\- output.txt

