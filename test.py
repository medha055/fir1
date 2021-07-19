import docx2txt
import PyPDF2
from nltk.tokenize import punkt
import pyresparser
import smtplib
import os
import spacy
spacy.load('en_core_web_sm')
from resume_parser import resumeparse
from pyresparser import ResumeParser
import pandas as pd
from sklearn import datasets
from pprint import pprint
import shutil
from pdfminer.high_level import extract_text
import os
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords 
import matplotlib.pyplot as plt



resume = docx2txt.process('/home/medha/Desktop/app2/resume screening/RecResume/lami.docx')
text_resume = str(resume)
print(summarize(text_resume,ratio=0.1))
# dir_list = os.listdir('/home/medha/Desktop/app2/resume screening/RecResume')
# #/home/medha/Desktop/app2/resume screening/RecResume/
# print("Total Resumes: ", len(dir_list))

#paths = r'/home/medha/Desktop/app2/resume screening/resumes'


resume_path = "/home/medha/Desktop/app2/resume screening/RecResume/"
accepted_path = "/home/medha/Desktop/app2/resume screening/resumes/Accepted/"
rejected_path = "/home/medha/Desktop/app2/resume screening/resumes/Rejected/"
# path = "C:\\Users\\prasa\\projects\\recruitment_project\\resumes\\pd.docx"



pdf_path = "/home/medha/Desktop/app2/resume screening/RecResume"
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)
    

text = extract_text_from_pdf('/home/medha/Desktop/app2/resume screening/RecResume/pd.pdf')
text = text.lower()
def p(self, b=None):
    if b is None:
        b = self.a
    print(b)
for sentence in self.text:
    sen=" ".join([words[0].lower9 for words in sentence])
    if re.search('experience',sen):
        sen_tokenised= nltk.word_tokenize(sen)
        tagged = nltk.pos_tag(sen_tokenised)
        entities = nltk.chunk.ne_chunk(tagged)
        for subtree in entities.subtrees():
            for leaf in subtree.leaves():
                if leaf[1] == 'CD':
                    experience=leaf[0]



# # #if __name__ == '__main__':
# #     print(extract_text_from_pdf('/home/medha/Desktop/app2/resume screening/RecResume/resume.pdf'))  # noqa: T001

# # def extract_text_from_docx(docx_path):
# #     txt = docx2txt.process(docx_path)
# #     if txt:
# #         return txt.replace('\t', ' ')
# #     return None


# # #if __name__ == '__main__':
# #     print(extract_text_from_docx('/home/medha/Desktop/app2/resume screening/RecResume/pd.docx'))

# # import docx2txt
# # import nltk
# # nltk.download(punkt)
# # nltk.download('stopwords')


# # # you may read the database from a csv file or some other database
# # SKILLS_DB = [
# #     'machine learning',
# #     'data science',
# #     'python',
# #     'word',
# #     'excel',
# #     'English',
# # ]


# # def extract_text_from_docx(docx_path):
# #     txt = docx2txt.process(docx_path)
# #     if txt:
# #         return txt.replace('\t', ' ')
# #     return None


# # def extract_skills(input_text):
# #     stop_words = set(nltk.corpus.stopwords.words('english'))
# #     word_tokens = nltk.tokenize.word_tokenize(input_text)

# #     # remove the stop words
# #     filtered_tokens = [w for w in word_tokens if w not in stop_words]

# #     # remove the punctuation
# #     filtered_tokens = [w for w in word_tokens if w.isalpha()]

# #     # generate bigrams and trigrams (such as artificial intelligence)
# #     bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

# #     # we create a set to keep the results in.
# #     found_skills = set()

# #     # we search for each token in our skills database
# #     for token in filtered_tokens:
# #         if token.lower() in SKILLS_DB:
# #             found_skills.add(token)

# #     # we search for each bigram and trigram in our skills database
# #     for ngram in bigrams_trigrams:
# #         if ngram.lower() in SKILLS_DB:
# #             found_skills.add(ngram)

# #     return found_skills
# # if __name__ == '__main__':
# #     text = extract_text_from_docx('/home/medha/Desktop/app2/resume screening/RecResume/pd.docx')
# #     skills = extract_skills(text)

# #     print(skills) 


    
terms = {'Operations management':['automation','bottleneck','constraints','cycle time','efficiency','fmea',
                                 'machinery','maintenance','manufacture','line balancing','oee','operations',
                                 'operations research','optimization','overall equipment effectiveness',
                                 'pfmea','process','process mapping','production','resources','safety',
                                 'stoppage','value stream mapping','utilization'],
        'Project management':['administration','agile','budget','cost','direction','feasibility analysis',
                              'finance','kanban','leader','leadership','management','milestones','planning',
                              'pmi','pmp','problem','project','risk','schedule','scrum','stakeholders'],
        'Soft skills':['communication','presentation skills','team building','leadership'],
        'Data analytics':['analytics','api','aws','big data','busines intelligence','clustering','code',
                          'coding','data','database','data mining','data science','deep learning','hadoop',
                          'hypothesis test','iot','internet','machine learning','modeling','nosql','nlp',
                          'predictive','programming','python','r','sql','tableau','text mining', 'MySQL','HTML',
                          'visualization']}

operations = 0
project = 0
data = 0
others = 0

scores =[]
for area in terms.keys():
    if area == 'Operations management':
        for word in terms[area]:
            if word in text:
                operations +=1
        scores.append(operations)
    
    elif area == 'Project management':
        for word in terms[area]:
            if word in text:
                project+=1
        scores.append(project)
    elif area == 'Soft skills':
        for word in terms[area]:
            if word in text:
                others+=1
        scores.append(others)
    else:
        for word in terms[area]:
            if word in text:
                data +=1
        scores.append(data)
summary = pd.DataFrame(scores,index=terms.keys(),columns=['score']).sort_values(by='score',ascending=False)  
print(summary)   
total = data + others + project + operations
print("The total score = " + str(total) )


pie = plt.figure(figsize=(10,10))
plt.pie(summary['score'], labels=summary.index, explode = (0.1,0,0,0), autopct='%1.0f%%',shadow=True,startangle=90)
plt.title('Python Developer Candidate - Resume Insight')
plt.axis('equal')
plt.show()

# Save pie chart as a .png file
pie.savefig('resume_insight.png')

#Salary prediction 



    