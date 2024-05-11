from transformers import pipeline

classifier = pipeline("sentiment-analysis")
a = classifier(
    [
        "do not go gentle into that good night",
        "nice to meet you"
    ]
)
print(a)