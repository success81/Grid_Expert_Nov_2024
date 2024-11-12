from flask import Flask, render_template, request, jsonify
import os
import random
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import markdown2
import PyPDF2
from werkzeug.utils import secure_filename
import io
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
PROJECT_ID = "winter-cogency-436501-g9"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

# Loading quotes
LOADING_QUOTES = [
    {"text": "Physics is the only profession in which prophecy is not only accurate but routine.", "author": "Neil deGrasse Tyson"},
    {"text": "If I have seen further, it is by standing on the shoulders of giants.", "author": "Isaac Newton"},
    {"text": "The most beautiful experience we can have is the mysterious.", "author": "Albert Einstein"},
    {"text": "Not only is the Universe stranger than we think, it is stranger than we can think.", "author": "Werner Heisenberg"},
    {"text": "The universe is under no obligation to make sense to you.", "author": "Neil deGrasse Tyson"},
    {"text": "I think nature's imagination is so much greater than man's, she's never going to let us relax.", "author": "Richard Feynman"},
    {"text": "Somewhere, something incredible is waiting to be known.", "author": "Carl Sagan"},
    {"text": "What I cannot create, I do not understand.", "author": "Richard Feynman"},
    {"text": "Energy cannot be created or destroyed; it can only be changed from one form to another.", "author": "Albert Einstein"},
    {"text": "The energy of the mind is the essence of life.", "author": "Aristotle"},
    {"text": "In physics, you don't have to go around making trouble for yourself - nature does it for you.", "author": "Frank Wilczek"},
    {"text": "The beauty of a living thing is not the atoms that go into it, but the way those atoms are put together.", "author": "Carl Sagan"},
    {"text": "The important thing is not to stop questioning. Curiosity has its own reason for existence.", "author": "Albert Einstein"},
    {"text": "Physics is really nothing more than a search for ultimate simplicity.", "author": "Richard Feynman"},
    {"text": "The good thing about science is that it's true whether or not you believe in it.", "author": "Neil deGrasse Tyson"}
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_text(pdf_file):
    """Extract text from PDF file"""
    pdf_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
    return pdf_text

def call_gemini_pro(prompt, temperature=1.0):
    """Call Gemini Pro model with given prompt"""
    try:
        model = GenerativeModel("gemini-1.5-pro-002")
        response = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 8000,
                "top_p": 1.0,
                "top_k": 40
            }
        )
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini: {str(e)}")
        raise
class ExpertListManager:
    def __init__(self):
        self.focused_experts = []
        self.creative_experts = []
        self.last_question = None

    def generate_lists(self, question, call_gemini_pro):
        """Generate fresh lists of experts if question changes"""
        if question != self.last_question:
            logger.info(f"Generating new expert lists for question: {question}")
            focused_prompt = f"""Create a diverse list of exactly 25 highly relevant experts for this question: "{question}"

Format each as:
- The [Title] with [X] Years in [Field]: [One-sentence specific expertise description]

Create 2-3 experts for each category:
1. Technical Implementation
2. Strategy and Planning
3. Risk Management
4. Ethical Considerations
5. Quality Assurance
6. User Experience
7. Data Management
8. Security
9. Compliance
10. Performance Optimization

Each expert must have:
- Different years of experience (10-40 years)
- Unique sub-specialty
- Distinct methodological approach
- Specific industry perspective
- Direct relevance to the problem

Make each expert highly specific and qualified."""

            creative_prompt = f"""Create a diverse list of exactly 25 unexpected experts for this question: "{question}"

Format each as:
- The [Title] with [X] Years in [Field/Domain]: [One-sentence specific expertise description]

Create 2-3 experts for each category:
1. Abstract Mathematical Concepts
2. Natural Phenomena
3. Historical Figures
4. Cultural Traditions
5. Arts and Crafts
6. Scientific Principles
7. Biological Systems
8. Philosophical Frameworks
9. Social Dynamics
10. Environmental Patterns

Each expert should:
- Represent different time periods
- Come from varied cultural contexts
- Offer unique philosophical approaches
- Apply unusual methodologies
- Bridge multiple disciplines

Make connections subtle and innovative."""

            try:
                focused_response = call_gemini_pro(focused_prompt, temperature=1.0)
                creative_response = call_gemini_pro(creative_prompt, temperature=1.0)
                
                self.focused_experts = [e.strip() for e in focused_response.split('\n') 
                                      if e.strip() and e.strip().startswith('-')]
                self.creative_experts = [e.strip() for e in creative_response.split('\n') 
                                       if e.strip() and e.strip().startswith('-')]
                
                logger.info(f"Generated {len(self.focused_experts)} focused experts and {len(self.creative_experts)} creative experts")
                
                self.last_question = question
            except Exception as e:
                logger.error(f"Error generating expert lists: {str(e)}")
                raise

    def get_experts(self, mode, num_experts):
        """Get the requested number of experts based on mode"""
        try:
            if mode == "focused":
                if not self.focused_experts:
                    logger.warning("No focused experts available")
                    return []
                return random.sample(self.focused_experts, min(num_experts, len(self.focused_experts)))
            elif mode == "creative":
                if not self.creative_experts:
                    logger.warning("No creative experts available")
                    return []
                return random.sample(self.creative_experts, min(num_experts, len(self.creative_experts)))
            else:  # mixed mode
                num_focused = num_experts // 2
                num_creative = num_experts - num_focused
                
                if not self.focused_experts or not self.creative_experts:
                    logger.warning("Missing expert lists for mixed mode")
                    return []
                
                focused = random.sample(self.focused_experts, min(num_focused, len(self.focused_experts)))
                creative = random.sample(self.creative_experts, min(num_creative, len(self.creative_experts)))
                return focused + creative
        except Exception as e:
            logger.error(f"Error selecting experts: {str(e)}")
            raise

expert_manager = ExpertListManager()
@app.route('/')
def index():
    return render_template('expert_grid.html')

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            pdf_text = extract_pdf_text(file)
            return jsonify({'text': pdf_text})
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        logger.info("Generating questions - request received")
        data = request.json
        main_question = data['question']
        additional_info = data.get('additional_info', '')
        pdf_content = data.get('pdf_content', '')

        context = f"""Additional Context: {additional_info}
        
Reference Material: {pdf_content}"""

        prompt = f"""Given this question/problem: "{main_question}"

Context:
{context}

Generate exactly 4 follow-up questions that will:
- Help thoroughly understand the context and requirements
- Cover different aspects of the problem
- Elicit specific, actionable information
- Provide necessary details for expert analysis

Format each question exactly as:
1- [Your question here]
2- [Your question here]
3- [Your question here]
4- [Your question here]

Make each question specific, clear, and directly relevant to solving the problem."""

        logger.info("Calling Gemini for question generation")
        response = call_gemini_pro(prompt, temperature=0.7)
        logger.info(f"Received response from Gemini: {response}")
        
        questions = [q.strip() for q in response.strip().split('\n') 
                    if q.strip() and any(q.strip().startswith(str(i) + '-') for i in range(1, 5))][:4]
        
        while len(questions) < 4:
            questions.append(f"{len(questions) + 1}- Please provide additional relevant information about this aspect.")
        
        return jsonify({
            'questions': questions,
            'loadingQuotes': LOADING_QUOTES
        })
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-advice', methods=['POST'])
def generate_advice():
    try:
        data = request.json
        main_question = data['question']
        mode = data['mode']
        additional_info = data.get('additional_info', '')
        answers = data.get('answers', {})
        num_experts = data.get('num_experts', 4)
        custom_experts = data.get('custom_experts', [])
        pdf_content = data.get('pdf_content', '')

        if mode == 'custom':
            if len(custom_experts) != num_experts:
                return jsonify({'error': f'Expected {num_experts} experts but got {len(custom_experts)}'}), 400
            selected_experts = custom_experts
        else:
            expert_manager.generate_lists(main_question, call_gemini_pro)
            selected_experts = expert_manager.get_experts(mode, num_experts)

        answers_text = "\n".join([f"Question {i+1}: {answers.get(str(i), '')}" for i in range(4)])

        if not selected_experts:
            selected_experts = [
                "- The Systems Analyst with 20 Years in Problem Solving: Specializes in breaking down complex challenges",
                "- The Implementation Expert with 15 Years in Project Execution: Masters at turning concepts into reality",
                "- The Risk Manager with 25 Years in Strategic Planning: Expert in identifying and mitigating potential issues",
                "- The Innovation Specialist with 18 Years in Creative Solutions: Focuses on novel approaches to challenges"
            ][:num_experts]

        simulation_messages = []
        for i in range(len(selected_experts)):
            simulation_messages.extend([
                f"Expert {i+1} analyzing problem space...",
                f"Expert {i+1} consulting domain expertise...",
                f"Expert {i+1} formulating recommendations...",
                f"Expert {i+1} finalizing analysis..."
            ])

        intro_prompt = f"""Create a comprehensive analysis of this question/problem:
"{main_question}"

Context Information:
{additional_info}

Follow-up Responses:
{answers_text}

Additional Reference Material:
{pdf_content}

Selected Experts:
{chr(10).join(selected_experts)}

Write an introduction that:
1. Frames the significance and complexity of the challenge
2. Introduces the assembled panel of experts
3. Previews the analysis structure

Write in an engaging, professional style.
Start with a creative, relevant title formatted in markdown."""

        introduction = call_gemini_pro(intro_prompt, temperature=0.9)

        expert_analyses = []
        for expert in selected_experts:
            expert_parts = expert.split(':')
            expert_title = expert_parts[0].strip('- ')
            expert_specialty = expert_parts[1].strip() if len(expert_parts) > 1 else ""

            expert_prompt = f"""As {expert_title}, with your specific expertise in {expert_specialty}, analyze this question:
"{main_question}"

Context:
{additional_info}

Follow-up Information:
{answers_text}

Additional Reference Material:
{pdf_content}

Create a detailed analysis that includes:

## Analysis from {expert_title}
- Your unique perspective based on your expertise
- Key principles and patterns you observe from your {expert_specialty} background
- Specific insights that only someone with your experience would notice
- Connections between your field and this challenge

## Expert Recommendations
- Concrete, actionable solutions drawing from your expertise
- Step-by-step implementation guidance using your field's methodologies
- Specific tools and techniques from your domain
- Innovative applications of your specialized knowledge
- Clear examples from your field of practice

## Implementation Strategy
- Detailed roadmap for executing your recommendations
- Specific milestones and timeline based on your experience
- Resource requirements and success metrics
- Change management approach from your perspective
- Critical success factors you've observed in your practice

## Risk Analysis and Mitigation
- Key risks identified from your expert viewpoint
- Early warning signs based on your experience
- Specific preventive measures from your field
- Contingency plans drawing from your expertise
- Long-term considerations from your domain perspective

Make all recommendations highly specific to your expertise.
Use concrete examples and terminology from your field.
Provide detailed, actionable advice that showcases your unique perspective.
Write in a style that reflects your years of experience in {expert_specialty}."""

            analysis = call_gemini_pro(expert_prompt, temperature=0.9)
            expert_analyses.append(analysis)

        conclusion_prompt = f"""Based on the expert analyses for this question about {main_question}, create a powerful two-paragraph conclusion that:

First Paragraph:
- Synthesize the key insights from all {len(selected_experts)} experts
- Highlight the most crucial and actionable recommendations
- Show how the different expert perspectives complement each other
- Emphasize the comprehensive nature of the solution
- Draw connections between the different expert viewpoints

Second Paragraph:
- Provide clear, prioritized next steps
- Offer a compelling vision of successful implementation
- Address potential challenges with a holistic perspective
- End with a strong call to action
- Emphasize the practical path forward

Make it both inspiring and practical.
Focus on actionable outcomes while maintaining strategic vision.
Show how the diverse expert perspectives come together into a coherent whole."""

        conclusion = call_gemini_pro(conclusion_prompt, temperature=0.9)

        full_analysis = f"""{introduction.strip()}

{chr(10).join(expert_analyses)}

## Conclusion
{conclusion.strip()}"""
        
        return jsonify({
            'html': markdown2.markdown(full_analysis),
            'markdown': full_analysis,
            'loadingQuotes': LOADING_QUOTES,
            'simulationMessages': simulation_messages
        })

    except Exception as e:
        logger.error(f"Error generating advice: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
