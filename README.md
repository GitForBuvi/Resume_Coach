# Resume_Coach

## Submission Details

- **Name:** Buvaneswari Anguchamy
- **Uplevel Email Address:** emailbuvana@gmail.com
- **Problem Statement:** Resume Coach

Resume Coach compares a resume with a target job description (JD), identifies
matching and missing skills, and suggests stronger experience bullets. The
Streamlit app also provides a benchmark page for evaluating prompt versions
against curated test datasets.

## How the application works

1. The candidate uploads a text-searchable PDF or DOCX resume and provides a
   job description by pasting text or entering a job-posting URL.
2. The app extracts resume text, checks the input, and sends it with the
   selected system prompt to the coaching model.
3. The app displays the match score, skills analysis, and suggested STAR
   rewrites, then runs an independent quality audit.
4. The candidate can download a tailored DOCX resume and a plain-text coaching
   report.
5. In the benchmark tab, the app evaluates a selected prompt against a golden
   dataset and logs aggregate results to MLflow.

## Project files

- [app.py](./app.py) contains the Streamlit interface and coordinates the
  analysis and benchmark workflows.
- [parser.py](./parser.py) extracts text from uploaded PDF and DOCX resumes.
- [security.py](./security.py) checks resume text and tracks daily usage.
- [coach.py](./coach.py) loads versioned prompts and requests the resume
  analysis.
- [doc_generator.py](./doc_generator.py) creates the tailored resume and report.
- [eval_judge.py](./eval_judge.py) audits generated coaching results.
- [evaluate.py](./evaluate.py) runs golden-dataset evaluations and logs metrics.
- [prompts/](./prompts) and [datasets/](./datasets) hold versioned prompt
  templates and evaluation data.
