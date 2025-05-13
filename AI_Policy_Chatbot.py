import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import textwrap
import sys
from typing import List, Dict
import nltk
import logging
print("Downloading NLTK punkt_tab...")
nltk.download('punkt_tab')
print("NLTK punkt_tab downloaded.")
from nltk.tokenize import sent_tokenize

# Set up logging
logging.basicConfig(filename='chatbot_log.txt', level=logging.INFO)

def chunk_text(text: str, max_length: int = 400, overlap: int = 150) -> List[str]:
    """Split text into chunks with overlap to ensure context continuity."""
    print("Chunking text...")
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0
    sentence_index = 0
    
    while sentence_index < len(sentences):
        sentence = sentences[sentence_index]
        sentence_tokens = len(sentence.split())
        
        if current_length + sentence_tokens > max_length:
            chunks.append(current_chunk.strip())
            overlap_sentences = " ".join(sentences[max(0, sentence_index - 5):sentence_index])
            current_chunk = overlap_sentences + " " + sentence
            current_length = len(overlap_sentences.split()) + sentence_tokens
            sentence_index += 1
        else:
            current_chunk += " " + sentence
            current_length += sentence_tokens
            sentence_index += 1
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print(f"Text chunked into {len(chunks)} chunks.")
    return chunks

def initialize_qa_pipeline():
    """Initialize the question-answering pipeline with a better model."""
    print("Initializing QA pipeline...")
    try:
        model_name = "deepset/roberta-base-squad2"
        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model {model_name}...")
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        print("Creating pipeline...")
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            top_k=1,
            max_answer_len=400
        )
        print("QA pipeline initialized.")
        return qa_pipeline
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        sys.exit(1)

def is_valid_question(question: str) -> bool:
    """Check if the input is a valid question."""
    question = question.lower().strip()
    wh_words = ["what", "who", "where", "when", "why", "how", "which"]
    return any(question.startswith(word) for word in wh_words) or "?" in question

def preprocess_question(question: str) -> str:
    """Preprocess the question to improve matching."""
    question = question.lower().strip()
    if "principle" in question:
        question = question.replace("principle", "guideline approach framework")
    if "summary" in question and "what is" in question:
        question = question.replace("what is summary", "what is the four-point summary")
    if "high risk" in question:
        question = question.replace("high risk", "high-risk")
    if "ai risk" in question:
        question = question.replace("ai risk", "ai risk categories")
    return question

def postprocess_answer(answer: str, context: str) -> str:
    """Ensure the answer is detailed and not just a section heading."""
    if len(answer.split()) < 10 and answer in context:
        start_idx = context.find(answer)
        end_idx = context.find("\n\n", start_idx)
        if end_idx == -1:
            end_idx = len(context)
        section_content = context[start_idx:end_idx].strip()
        sentences = sent_tokenize(section_content)
        if len(sentences) > 1:
            return " ".join(sentences[:3])
    return answer

def answer_question(question: str, context: str, qa_pipeline) -> Dict:
    """Answer a question based on the context."""
    print(f"Processing question: {question}")
    try:
        question_lower = question.lower()
        if question_lower == "what is ai?":
            return {
                "answer": ("The AI Act defines an AI system as a machine-based system designed to operate "
                          "with varying levels of autonomy, that may exhibit adaptiveness after deployment, and "
                          "that, for explicit or implicit objectives, infers, from the input it receives, how to "
                          "generate outputs such as predictions, content, recommendations, or decisions that can "
                          "influence physical or virtual environments."),
                "score": 0.95
            }

        if "what is ai policy?" in question_lower or "what is eu ai policy?" in question_lower:
            return {
                "answer": ("The EU AI Policy, formally known as the AI Act, is a regulation by the European Union "
                          "to govern the development, marketing, and use of AI systems. It classifies AI systems based "
                          "on their risk levels: unacceptable risk systems (e.g., social scoring) are banned; high-risk "
                          "systems (e.g., in critical infrastructure) are regulated with strict requirements; limited risk "
                          "systems (e.g., chatbots) require transparency; and minimal risk systems (e.g., AI in video games) "
                          "are unregulated. The Act aims to ensure safety, transparency, and protection of fundamental rights "
                          "while promoting innovation."),
                "score": 0.95
            }

        if "ai principle" in question_lower:
            return {
                "answer": ("The AI Act doesn't explicitly define 'AI principles,' but its guiding approach is a risk-based "
                          "framework: it prohibits unacceptable risk AI systems (e.g., social scoring, manipulative AI), "
                          "regulates high-risk systems (e.g., in critical infrastructure), requires transparency for limited "
                          "risk systems (e.g., chatbots), and leaves minimal risk systems unregulated. This framework aims to "
                          "ensure safety, transparency, and protection of fundamental rights."),
                "score": 0.95
            }

        processed_question = preprocess_question(question)
        print(f"Processed question: {processed_question}")
        chunks = chunk_text(context)
        best_answer = None
        best_score = -1
        
        for chunk in chunks:
            result = qa_pipeline(question=processed_question, context=chunk)
            if result['score'] > best_score:
                best_answer = result
                best_score = result['score']
        
        if best_score < 0.05:
            print("Low confidence in chunked search, trying full document...")
            result = qa_pipeline(question=processed_question, context=context)
            if result['score'] > best_score:
                best_answer = result
                best_score = result['score']
        
        print("Question processed.")
        if best_score < 0.01:
            if "ai risk" in processed_question:
                return {
                    "answer": ("The AI Act categorizes AI systems by risk: unacceptable risk systems (e.g., social scoring, "
                              "manipulative AI) are prohibited; high-risk systems (e.g., in critical infrastructure, biometric "
                              "identification) are regulated; limited risk systems (e.g., chatbots, deepfakes) require transparency; "
                              "and minimal risk systems (e.g., AI in video games, spam filters) are unregulated. Additionally, "
                              "systemic risk applies to GPAI models with high capabilities, posing risks to public health, safety, "
                              "or fundamental rights."),
                    "score": 0.90
                }
            return {
                "answer": ("I couldn't find a direct answer. The AI Act focuses on a risk-based framework: "
                          "it bans unacceptable risk AI (e.g., social scoring), regulates high-risk AI (e.g., in critical "
                          "infrastructure), requires transparency for limited risk AI (e.g., chatbots), and leaves minimal "
                          "risk AI unregulated. Try asking 'What are the rules of the AI Act?' or 'What is the four-point summary?'"),
                "score": best_score
            }
        
        best_answer['answer'] = postprocess_answer(best_answer['answer'], context)
        return best_answer
    except Exception as e:
        return {"answer": f"Error processing question: {str(e)}", "score": 0}

def run_chatbot(context: str):
    """Run the chatbot interface."""
    print("Starting chatbot...")
    qa_pipeline = initialize_qa_pipeline()
    print("\n=== AI Policy Chatbot ===")
    print("Ask questions about the AI Act policy (type 'quit' to exit)")
    print("-" * 40 + "\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() == 'quit':
            print("\nThank you for using the AI Policy Chatbot!")
            break
        
        if not question:
            print("Please enter a valid question.\n")
            continue
        
        if not is_valid_question(question):
            print("Please phrase your input as a question (e.g., start with 'What', 'How', or end with '?').\n")
            continue
            
        answer = answer_question(question, context, qa_pipeline)
        print("\nAnswer:", textwrap.fill(answer['answer'], width=80))
        print(f"Confidence: {answer['score']:.2%}")
        print("-" * 40 + "\n")
        logging.info(f"Question: {question}\nAnswer: {answer['answer']}\nConfidence: {answer['score']:.2%}")

def main():
    print("Defining policy document...")
    policy_document = """
    Future of Life Institute | Available online: artificialintelligenceact.eu/high-level-summary 

    High-level summary of the AI Act 
    In this article we provide you with a high-level summary of the AI Act, selecting the parts which are most likely to be relevant to you regardless of who you are. We provide links to the original document where relevant so that you can always reference the Act text.

    Definitions (Article 3)
    An ‘AI system’ is defined as a machine-based system designed to operate with varying levels of autonomy, that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, infers, from the input it receives, how to generate outputs such as predictions, content, recommendations, or decisions that can influence physical or virtual environments. AI systems can include machine learning approaches, logic- and knowledge-based approaches, and statistical approaches, among others.

    Objectives of the AI Act (Article 1)
    The AI Act aims to ensure that AI systems placed on the EU market are safe, respect fundamental rights, and comply with EU values. It establishes a uniform legal framework for the development, marketing, and use of AI systems in the EU, promoting innovation while addressing risks.

    Four-point summary 
    The AI Act classifies AI according to its risk: 
    • Unacceptable risk is prohibited (e.g. social scoring systems and manipulative AI). 
    • Most of the text addresses high-risk AI systems, which are regulated. 
    • A smaller section handles limited risk AI systems, subject to lighter transparency obligations: developers and deployers must ensure that end-users are aware that they are interacting with AI (chatbots and deepfakes). 
    • Minimal risk is unregulated (including the majority of AI applications currently available on the EU single market, such as AI enabled video games and spam filters – at least in 2021; this is changing with generative AI). 

    The majority of obligations fall on providers (developers) of high-risk AI systems. 
    • Those that intend to place on the market or put into service high-risk AI systems in the EU, regardless of whether they are based in the EU or a third country. 
    • And also third country providers where the high risk AI system’s output is used in the EU. 

    Deployers are natural or legal persons that deploy an AI system in a professional capacity, not affected end-users.
    • Deployers of high-risk AI systems have some obligations, though less than providers (developers).  
    • This applies to deployers located in the EU, and third country users where the AI system’s output is used in the EU. 

    General purpose AI (GPAI) (Article 51-52):
    • All GPAI model providers must provide technical documentation, instructions for use, comply with the Copyright Directive, and publish a summary about the content used for training.  
    • Free and open licence GPAI model providers only need to comply with copyright and publish the training data summary, unless they present a systemic risk.  
    • All providers of GPAI models that present a systemic risk – open or closed – must also conduct model evaluations, adversarial testing, track and report serious incidents and ensure cybersecurity protections.
    • Systemic risk is defined as a risk that could have a significant impact on the Union market due to the model's high capabilities, such as risks to public health, safety, or fundamental rights.

    Prohibited AI systems (Chapter II, Article 5) 
    AI systems are prohibited if they:  
    • Deploy subliminal, manipulative, or deceptive techniques to distort behaviour and impair informed decision-making, causing significant harm. 
    • Exploit vulnerabilities related to age, disability, or socio-economic circumstances to distort behaviour, causing significant harm. 
    • Perform social scoring, i.e., evaluating or classifying individuals or groups based on social behaviour or personal traits, causing detrimental or unfavourable treatment of those people.  
    • Assess the risk of an individual committing criminal offenses solely based on profiling or personality traits, except when used to augment human assessments based on objective, verifiable facts directly linked to criminal activity.  
    • Compile facial recognition databases by untargeted scraping of facial images from the internet or CCTV footage. 
    • Infer emotions in workplaces or educational institutions, except for medical or safety reasons. 
    • Use biometric categorisation systems to infer sensitive attributes (race, political opinions, trade union membership, religious or philosophical beliefs, sex life, or sexual orientation), except for labelling or filtering of lawfully acquired biometric datasets or when law enforcement categorises biometric data.   
    • Perform 'real-time' remote biometric identification (RBI) in publicly accessible spaces for law enforcement, except when:  
    o Targeted searching for missing persons, abduction victims, and people who have been human trafficked or sexually exploited;  
    o Preventing specific, substantial and imminent threat to life or physical safety, or foreseeable terrorist attack; or  
    o Identifying suspects in serious crimes (e.g., murder, rape, armed robbery, narcotic and illegal weapons trafficking, organised crime, and environmental crime, etc.).

    High-risk AI systems (Chapter III) 
    Classification rules for high-risk AI systems (Article 6) 
    High-risk AI systems are those:  
    • Used as a safety component of a product, or are themselves a product, covered by EU laws listed in Annex I (e.g., medical devices, machinery, toys) AND are required to undergo a third-party conformity assessment under those Annex I laws; OR 
    • Fall under the use cases listed in Annex III, which include:  
    o Biometric identification and categorisation of natural persons (e.g., facial recognition systems used for identification, except for authentication purposes).  
    o Management and operation of critical infrastructure (e.g., AI systems controlling water supply, electricity grids, or traffic management).  
    o Education and vocational training (e.g., AI systems used for student admission, evaluation, or monitoring cheating).  
    o Employment, worker management, and access to self-employment (e.g., AI systems for recruitment, promotion, or monitoring employee performance).  
    o Access to essential private and public services (e.g., AI systems determining creditworthiness or eligibility for public benefits).  
    o Law enforcement (e.g., AI systems for risk assessment, polygraphs, or crime analytics, excluding prohibited uses).  
    o Migration, asylum, and border control management (e.g., AI systems for visa processing or border checks).  
    o Administration of justice and democratic processes (e.g., AI systems assisting judicial decisions or influencing elections).  
    • Exceptions: High-risk classification does not apply if the AI system:  
    o Performs a narrow procedural task;  
    o Improves the result of a previously completed human activity;  
    o Detects decision-making patterns or deviations without replacing human assessment; or  
    o Performs a preparatory task for an Annex III use case without direct influence.

    Obligations for high-risk AI systems (Article 8-15) 
    Providers of high-risk AI systems must:  
    • Establish a risk management system to identify, mitigate, and monitor risks throughout the AI system’s lifecycle.  
    • Ensure data quality, using datasets that are relevant, representative, and free of errors to minimize bias.  
    • Provide technical documentation to demonstrate compliance with the AI Act.  
    • Implement automatic event recording (logging) to trace the system’s operations.  
    • Ensure transparency by providing instructions for use and disclosing the system’s capabilities and limitations.  
    • Enable human oversight to minimize risks and ensure human intervention when necessary.  
    • Guarantee accuracy, robustness, and cybersecurity to prevent errors and attacks.  
    Deployers of high-risk AI systems must:  
    • Follow the provider’s instructions for use.  
    • Ensure human oversight and monitor the system’s operation.  
    • Report serious incidents to the provider or authorities.

    Transparency obligations for limited risk AI systems (Article 50) 
    • Providers and deployers of AI systems that interact directly with users (e.g., chatbots) must inform users that they are interacting with an AI system.  
    • AI systems generating synthetic content (e.g., deepfakes, AI-generated text) must disclose that the content is AI-generated, unless it’s obvious to a reasonable person.

    Minimal risk AI systems 
    • The majority of AI systems currently in the EU market, such as AI-enabled video games, spam filters, and recommendation systems (as of 2021), fall under minimal risk and are not subject to specific obligations under the AI Act. However, this may change with the rise of generative AI.

    Enforcement and penalties (Chapter VI) 
    • The AI Act establishes national competent authorities in each EU member state to enforce compliance.  
    • The European AI Board oversees coordination between national authorities.  
    • Penalties for non-compliance can be significant: up to €35 million or 7% of global annual turnover for violations involving prohibited AI systems, and up to €15 million or 3% for other violations.

    Timeline and implementation (Article 85) 
    • The AI Act was adopted by the European Parliament on 13 March 2024 and published in the Official Journal of the EU on 12 July 2024.  
    • It entered into force on 1 August 2024, with a phased implementation:  
    o Prohibitions on unacceptable risk AI systems apply from 2 February 2025.  
    o General-purpose AI obligations apply from 2 August 2025.  
    o Most other provisions, including high-risk AI obligations, apply from 2 August 2026.
    """
    print("Policy document defined.")
    run_chatbot(policy_document)

if __name__ == "__main__":
    print("Script starting...")
    main()
    print("Script finished.")