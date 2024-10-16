# Imports the Google Cloud client library
from google.cloud import language_v1beta2 as language_v1

# from google.cloud import language_v2
from google.cloud.language_v1beta2.types import ClassificationModelOptions
# Instantiates a client
client = language_v1.LanguageServiceClient()

# The text to analyze
text = "Russell Hopton acted in many films until his death in 1945. He only directed 2 and \"Black Gold\" was one, (the other was also from 1936). Frankie Darro had a sometimes abrasive screen presence but in this he was playing a good kid. He was obviously quite popular on the \"quickie\" circuit - he made so many films. In this one he plays the son of an old oil rigger who is convinced that he will strike oil very soon.<br /><br />J.C. Anderson (Berton Churchill) is trying to convince the old man to sell up as he knows there is going to be oil struck at any moment. A geologist, Henry, comes on the scene and helps \"Fishtail's\" dad. He also convinces \"Fishtail\" to go to school regularly. Henry has his eye on Cynthia, the pretty teacher. This was Gloria Shea's last film - she had begun her career as Olive Shea in \"Glorifying the American Girl\" (1929). \"Fishtail's\" dad is killed when the rig is sabotaged and Henry is determined to bring Anderson and his cronies to justice. When Henry is kidnapped Anderson tries to persuade \"Fishtail\" to sell his oil lease. It all ends well with oil being struck and \"Fishtail\" going to Military school.<br /><br />It is okay for a rainy day."
document = language_v1.Document(
    content=text, type_=language_v1.Document.Type.PLAIN_TEXT
)


# Detects the sentiment of the text
sentiment = client.classify_text(
    request={"document": document,"classification_model_options":ClassificationModelOptions.V2Model}
)

print("Text: {}".format(text))
print("Sentiment: {}, {}".format(sentiment.score, sentiment.magnitude))