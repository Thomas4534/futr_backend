from flask import Flask, render_template, request, jsonify
from markupsafe import Markup
import random
import json
import html
from fake_useragent import UserAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
import re
import time

app = Flask(__name__)


THRESHOLD = 0.55
ua = UserAgent()


edu_rank = {
        "no education": 0,
        "certificate": 1,
        "associate": 2,
        "bachelor": 3,
        "master": 4,
        "phd": 4
    }

edu_normalize = {
        "No education requirement mentioned.": "no education",
        "Certificate required.": "certificate",
        "Certificate mentioned.": "certificate",
        "Associate degree required.": "associate",
        "Associate degree mentioned.": "associate",
        "Bachelor's degree required.": "bachelor",
        "Bachelor's degree mentioned.": "bachelor",
        "Bachelor's degree preferred.": "bachelor",
        "Master's degree required.": "master",
        "Master's degree preferred.": "master",
        "Master's degree mentioned.": "master",
        "PhD required.": "phd",
        "PhD mentioned.": "phd"
    }

@app.route("/", methods=["GET", "POST"])
def index():
    start_total = time.perf_counter()
    jobs = []
    user_skills = []

    if request.method == "POST":
        tags_json = request.form.get("tags_json", "[]")
        try:
            tags = json.loads(tags_json)
        except:
            tags = []

        min_salary = int(request.form.get("min_salary", 0))
        user_education = request.form.get("education_level", "no education")

        skills_json = request.form.get("skills_json", "[]")
        try:
            user_skills = json.loads(skills_json)
        except:
            user_skills = []


        cities_states, positions = [], []
        for tag in tags:
            if "," in tag:
                parts = tag.split(",")
                if len(parts) == 2:
                    cities_states.append((parts[0].strip(), parts[1].strip()))
            else:
                positions.append(tag.strip())

        if not cities_states:
            cities_states = [("", "")]
        if not positions:
            positions = [""]

        from itertools import product
        combinations = list(product(cities_states, positions))
        random.shuffle(combinations)

        all_jobs = []
        seen_links = set()
        max_total_jobs = 10
        max_pages = 3
        page = 0
        finished_combos = set()

        while len(all_jobs) < max_total_jobs and len(finished_combos) < len(combinations):
            for idx, (city_state, position) in enumerate(combinations):
                if idx in finished_combos:
                    continue
                city, state = city_state
                scrape_start = time.perf_counter()
                jobs_batch = scrape_linkedin(min_salary, city, state, page, position)
                scrape_end = time.perf_counter()
                print(f"[DEBUG] scrape_linkedin() took {scrape_end - scrape_start:.2f} sec")
                if not jobs_batch:
                    finished_combos.add(idx)
                    continue
                added_this_round = 0
                jobs_processing_start = time.perf_counter()
                for job in jobs_batch:
                    if job['url'] not in seen_links:
                        all_jobs.append(job)
                        seen_links.add(job['url'])
                        added_this_round += 1
                        if len(all_jobs) >= max_total_jobs:
                            break
                jobs_processing_end = time.perf_counter()
                print(f"[DEBUG] Jobs processing took {jobs_processing_end - jobs_processing_start:.2f} sec")
                if added_this_round == 0 or page >= max_pages:
                    finished_combos.add(idx)
                if len(all_jobs) >= max_total_jobs:
                    break
            page += 1

        user_rank = edu_rank.get(user_education, 0)

        jobs = [
            job for job in all_jobs
            if edu_rank.get(
                edu_normalize.get(job.get("education_requirement", "No education requirement mentioned."),"no education"),
                0
            ) <= user_rank
        ]

        for job in jobs:
            if 'job_details' in job:
                jd_html = job['job_details']
                job['job_details'] = Markup(format_job_description(jd_html))
                jd_plain_text = Markup(jd_html).striptags()

                try:

                    mistral_start = time.perf_counter()
                    job_skills = extract_skills_from_text(jd_plain_text)
                    mistral_end = time.perf_counter()
                    print(
                        f"[DEBUG] Mistral extraction took {mistral_end - mistral_start:.2f} sec for job '{job.get('title')}'")

                    job["extracted_skills"] = job_skills

                    if user_skills:
                        avg_score = compare_skill_lists(job_skills, user_skills)
                        job["score"] = round(avg_score)
                    else:
                        job["score"] = None


                except Exception as e:
                    print(f"[ERROR] Exception occurred: {e}")
                    job["extracted_skills"] = []
                    job["score"] = None

    end_total = time.perf_counter()
    print(f"[DEBUG] TOTAL request time: {end_total - start_total:.2f} sec")

    return render_template(
        "index.html",
        jobs=jobs,
        tags_json=request.form.get("tags_json", "[]"),
        min_salary=int(request.form.get("min_salary", 0)),
        user_skills=user_skills,
        skills_input_value=user_skills,
        education_level=request.form.get("education_level", "no education")
    )

@app.route("/load_more", methods=["GET"])
def load_more():
    city = request.args.get("city", "").strip()
    state = request.args.get("state", "").strip()
    position = request.args.get("position", "").strip()
    min_salary = int(request.args.get("min_salary", 0))
    page = int(request.args.get("page", 0))

    skills_json = request.args.get("skills_json", "[]")
    try:
        user_skills = json.loads(skills_json)
    except:
        user_skills = []

    user_education = request.args.get("education_level", "no education")

    # Precompute embeddings for user skills

    jobs = scrape_linkedin(min_salary, city, state, page, position)

    user_rank = edu_rank.get(user_education, 0)

    # Filter jobs by education requirement
    filtered_jobs = [
        job for job in jobs
        if edu_rank.get(
            edu_normalize.get(job.get("education_requirement", "No education requirement mentioned."), "no education"),
            0
        ) <= user_rank
    ]

    for job in filtered_jobs:
        job['job_details'] = format_job_description(job['job_details'])
        try:
            jd_plain_text = Markup(job['job_details']).striptags()
            job_skills = extract_skills_from_text(jd_plain_text)
            job["extracted_skills"] = job_skills
            if user_skills:
                avg_score = compare_skill_lists(job_skills, user_skills)
                job["score"] = int(round(avg_score))
            else:
                job["score"] = None
        except:
            job["extracted_skills"] = []
            job["score"] = None

    return jsonify(filtered_jobs)




def get_headers():
    return {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1'
    }


def format_job_description(text):
    text = html.unescape(text)
    text = re.sub(r"Show more.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n+", "\n", text).strip()
    soup = BeautifulSoup(text, 'html.parser')

    for strong_tag in soup.find_all('strong'):
        strong_tag.insert_before('<span style="font-weight: bold;">')
        strong_tag.insert_after('</span>')
        strong_tag.unwrap()

    formatted_text = str(soup)
    formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text).strip()

    # Adjust trimming as needed
    formatted_text = formatted_text[:-55]
    return formatted_text


def fetch_salary_from_job_page(job_url):
    try:
        response = requests.get(job_url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # JSON-LD salary extraction
        script_tags = soup.find_all('script', type='application/ld+json')
        for script in script_tags:
            try:
                data = json.loads(script.string)
                if data.get('@type') == 'JobPosting' and 'baseSalary' in data:
                    base_salary = data['baseSalary']
                    currency = base_salary.get('currency', 'USD')
                    value = base_salary.get('value')

                    min_salary = None
                    max_salary = None

                    if isinstance(value, dict):
                        min_salary = value.get('minValue') or value.get('value')
                        max_salary = value.get('maxValue')
                    elif isinstance(value, (int, float)):
                        min_salary = value
                    elif isinstance(value, str):
                        try:
                            min_salary = float(value.replace(',', '').strip())
                        except:
                            pass

                    unit_str = 'year' if min_salary and min_salary >= 1000 else 'hour'

                    if min_salary and max_salary:
                        if min_salary == max_salary:
                            return f"{currency} {min_salary:,.2f} / {unit_str}"
                        else:
                            return f"{currency} {min_salary:,.2f} - {max_salary:,.2f} / {unit_str}"
                    elif min_salary:
                        return f"{currency} {min_salary:,.2f} / {unit_str}"
            except Exception:
                continue

        # Fallback: look for text patterns like "$xx / hr"
        salary_text = soup.find_all('span', string=re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{1,2})?\s?/hr', re.IGNORECASE))
        for salary in salary_text:
            salary_value = salary.get_text().strip()
            if salary_value:
                salary_value = salary_value.replace('$', '').replace('/hr', '').strip()
                try:
                    salary_value = float(salary_value.replace(',', ''))
                    return f"USD {salary_value:,.2f} / hour"
                except ValueError:
                    pass
        return "Salary not specified"

    except Exception as e:
        print(f"Error scraping salary from {job_url}: {e}")
        return "Salary not specified"

def fetch_job_details_from_job_page(job_url):
    try:
        response = requests.get(job_url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        about_section = soup.find('section', {'class': 'description'})
        return str(about_section) if about_section else "No detailed description available."
    except Exception as e:
        print(f"Error scraping job details from {job_url}: {e}")
        return "No detailed description available."


def fetch_salary_and_details(link):
    try:
        salary = fetch_salary_from_job_page(link)
        job_details_raw = fetch_job_details_from_job_page(link)
        job_details_text = BeautifulSoup(job_details_raw, 'html.parser').get_text(separator=' ', strip=True)
        return salary, job_details_raw, job_details_text
    except Exception:
        return "Not specified", "No detailed description available.", ""




def scrape_linkedin(min_salary, city="", state="", page=0, position=""):
    scrape_total_start = time.perf_counter()
    jobs = []

    try:
        location_query = ""
        if city and state:
            location_query = f"{city}, {state}".replace(" ", "%20")
        elif city:
            location_query = city.replace(" ", "%20")
        elif state:
            location_query = state.replace(" ", "%20")

        keywords_param = position.replace(" ", "%20")
        start = page * 25

        url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?"
        params = []
        if keywords_param:
            params.append(f"keywords={keywords_param}")
        if location_query:
            params.append(f"location={location_query}")
        params.append(f"start={start}")
        url += "&".join(params)

        response = requests.get(url, headers=get_headers(), timeout=15)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        job_elements = soup.find_all('li')

        basic_jobs = []
        for job in job_elements:
            try:
                title_elem = job.find('h3', class_='base-search-card__title')
                company_elem = job.find('h4', class_='base-search-card__subtitle')
                location_elem = job.find('span', class_='job-search-card__location')
                link_elem = job.find('a', class_='base-card__full-link')

                if not all([title_elem, company_elem, location_elem, link_elem]):
                    continue

                job_url = link_elem['href'].split('?')[0]
                basic_jobs.append({
                    'title': title_elem.text.strip(),
                    'company': company_elem.text.strip(),
                    'location': location_elem.text.strip(),
                    'url': job_url
                })
            except:
                continue

        jobs_with_details = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_job = {executor.submit(fetch_salary_and_details, job['url']): job for job in basic_jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    salary, job_details_raw, job_details_text = future.result()
                except Exception:
                    salary, job_details_raw, job_details_text = "Not specified", "No detailed description available.", ""

                include_job = True
                if "Not specified" not in salary and re.search(r"\d", salary):
                    numbers = [int(num.replace(",", "")) for num in re.findall(r"\d[\d,]*", salary)]
                    if len(numbers) >= 2:
                        high = max(numbers)
                        if high < min_salary:
                            include_job = False
                    elif len(numbers) == 1 and numbers[0] < min_salary:
                        include_job = False

                if not include_job:
                    continue

                education_requirement = detect_education_requirement(job_details_text)
                visa_requirement = detect_visa_requirement(job_details_text)

                jobs_with_details.append({
                    'title': job['title'],
                    'company': job['company'],
                    'location': job['location'],
                    'salary': salary,
                    'url': job['url'],
                    'job_details': format_job_description(job_details_raw),
                    'job_details_text': job_details_text,
                    'education_requirement': education_requirement,
                    'visa_requirement': visa_requirement,
                    'source': 'LinkedIn'
                })

        scrape_total_end = time.perf_counter()
        print(f"[DEBUG] scrape_linkedin() TOTAL: {scrape_total_end - scrape_total_start:.2f} sec")
        return jobs_with_details

    except Exception as e:
        print(f"Exception in scrape_linkedin: {e}")
        return []




# Helper: Normalize text
def normalize_text(text):
    return text.lower().replace("’", "'").replace("“", '"').replace("”", '"')

# Education negation patterns (explicitly say "no degree required")
education_negation_patterns = [
    re.compile(p) for p in [
        r"no (degree|bachelor'?s|master'?s|ph\.?d\.?|education) required",
        r"(degree|education) (is not|not) (required|needed|mandatory|necessary)",
        r"(without|does not require|don't require|doesn'?t need) (a )?(degree|education|bachelor'?s|master'?s)"
    ]
]

# Soft education patterns (preferred but not required)
education_soft_patterns = [
    ("Certificate preferred.", re.compile(r"\b(certificate|certification|diploma)\b.*?(preferred|desired|nice to have)")),
    ("Associate degree preferred.", re.compile(r"\b(associate('|s)? degree|associate)\b.*?(preferred|desired|nice to have)")),
    ("Bachelor's degree preferred.", re.compile(r"\b(bachelor('|s)?|b\.a\.|b\.s\.|bsc)\b.*?(preferred|desired|nice to have)")),
    ("Master's degree preferred.", re.compile(r"\b(master('|s)?|m\.a\.|m\.s\.|msc)\b.*?(preferred|desired|nice to have)")),
    ("PhD preferred.", re.compile(r"\b(ph\.?d\.?|doctorate|doctoral)\b.*?(preferred|desired|nice to have)")),
]

# Hard education levels (required)
education_levels = [
    ("Certificate required.", re.compile(r"\b(certificate|certification|diploma)\b.*?(required|mandatory|must)")),
    ("Associate degree required.", re.compile(r"\b(associate('|s)? degree|associate)\b.*?(required|mandatory|must)")),
    ("Bachelor's degree required.", re.compile(r"\b(bachelor('|s)?|b\.a\.|b\.s\.|bsc)\b.*?(required|mandatory|must)")),
    ("Master's degree required.", re.compile(r"\b(master('|s)?|m\.a\.|m\.s\.|msc)\b.*?(required|mandatory|must)")),
    ("PhD required.", re.compile(r"\b(ph\.?d\.?|doctorate|doctoral)\b.*?(required|mandatory|must)")),
]

# Fallback detection (no keyword like "required" but a degree is mentioned)
education_fallback = [
    ("Certificate mentioned.", re.compile(r"\b(certificate|certification|diploma)\b")),
    ("Associate degree mentioned.", re.compile(r"\b(associate('|s)? degree|associate)\b")),
    ("Bachelor's degree mentioned.", re.compile(r"\b(bachelor('|s)?|b\.a\.|b\.s\.|bsc)\b")),
    ("Master's degree mentioned.", re.compile(r"\b(master('|s)?|m\.a\.|m\.s\.|msc)\b")),
    ("PhD mentioned.", re.compile(r"\b(ph\.?d\.?|doctorate|doctoral)\b")),
]

def detect_education_requirement(text):
    text = normalize_text(text)

    for pattern in education_negation_patterns:
        if pattern.search(text):
            return "No education requirement mentioned."

    for label, pattern in education_levels:
        if pattern.search(text):
            return label

    for label, pattern in education_soft_patterns:
        if pattern.search(text):
            return label

    for label, pattern in education_fallback:
        if pattern.search(text):
            return label

    return "No education requirement mentioned."


# Visa negation patterns
visa_negation_patterns = [
    re.compile(p) for p in [
        r"no (visa|sponsorship|citizenship) required",
        r"(visa|sponsorship|citizenship) (is not|not) (required|provided|available)",
        r"(must|should) (be able to work) (without|independently of) (visa|sponsorship)",
        r"authorized to work.*?no sponsorship",
        r"does not sponsor"
    ]
]

# Hard visa requirements
visa_requirements = [
    ("U.S. citizenship required.", re.compile(r"\b(us citizen|u\.s\. citizen|citizenship required|must be a citizen|security clearance)\b")),
    ("Green card required.", re.compile(r"\b(green card|permanent resident|lawful permanent resident|lpr)\b")),
    ("Eligibility to work in the U.S. required.", re.compile(
        r"\b(work authorization|authorized to work in the us|legally work in the us|valid work permit"
        r"|must be eligible to work in the us|eligibility to work in the us|eligible to work in the us)\b")),
    ("H1B sponsorship available.", re.compile(r"\b(h[- ]?1b|h1b|sponsorship available|we sponsor|visa sponsorship)\b")),
]

# Soft visa mentions
visa_soft_requirements = [
    ("Sponsorship may be available.", re.compile(r"(may offer|can offer|might provide).*?(visa|sponsorship)")),
    ("Citizenship or work eligibility preferred.", re.compile(r"(citizenship|work authorization).*(preferred|desired|ideally)")),
]

def detect_visa_requirement(text):
    text = normalize_text(text)

    for pattern in visa_negation_patterns:
        if pattern.search(text):
            return "No visa or citizenship requirement mentioned."

    for label, pattern in visa_requirements:
        if pattern.search(text):
            return label

    for label, pattern in visa_soft_requirements:
        if pattern.search(text):
            return label

    return "No visa or citizenship requirement mentioned."


#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------FIND SKILLS-------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

from collections import defaultdict

import spacy
import subprocess
import sys

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_skills_from_text(text):
    doc = nlp(text)

    keywords = []
    seen = set()
    for token in doc:
        if token.pos_ in {"NOUN", "VERB", "ADJ"}:
            word = token.lemma_.lower().strip()
            if len(word) > 2 and word.isalpha() and not token.is_stop:
                if word not in seen:
                    keywords.append(word)
                    seen.add(word)

    # Step 2: Add noun chunks (e.g., "software installation")
    keywords += [
        chunk.lemma_.lower().replace(" ", "_")
        for chunk in doc.noun_chunks
        if len(chunk) > 1 and not any(t.is_stop for t in chunk)
    ]

    # Step 3: Filter by frequency (keep nouns appearing ≥1 time)
    noun_freq = defaultdict(int)
    for token in doc:
        if token.pos_ == "NOUN":
            noun_freq[token.lemma_.lower()] += 1

    keywords = [
        word for word in keywords
        if (noun_freq.get(word.split("_")[0], 0) >= 1)
    ]

    # Filter keywords to keep only the most job-specific ones (7-8 max)
    def filter_keywords(keywords, doc):
        scores = {}
        for term in keywords:
            # Boost terms in "Required Skills" section
            in_skills_section = any(
                "required skills" in sent.text.lower() and term in sent.text.lower()
                for sent in doc.sents
            )

            # Score components
            freq_score = sum(1 for token in doc if token.lemma_.lower() in term)
            length_score = len(term.split("_"))  # Prefer multi-word terms
            position_score = 5 if in_skills_section else 1

            scores[term] = freq_score * length_score * position_score

        return sorted(scores.keys(), key=lambda x: -scores[x])[:8]

    def clean_keyword(term):
        term = term.replace("\n", " ")
        term = term.replace("_", " ")
        term = " ".join(term.split())
        term = term.title()
        return term

    filtered_skills = [
        clean_keyword(k)
        for k in filter_keywords(keywords, doc)
        if len(clean_keyword(k).split()) > 1
    ]

    return filtered_skills

#-----------------------------------------------------------------------------------------------------------------------
#--------------------------------------CALCULATE SCORE------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compare_skill_lists(job_skills, user_skills):
    results = []
    scores = []

    # Encode and normalize embeddings
    def normalize(vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    job_embeddings = [normalize(nlp(skill).vector) for skill in job_skills]
    user_embeddings = [normalize(nlp(skill).vector) for skill in user_skills]

    for j_idx, js in enumerate(job_skills):
        for u_idx, us in enumerate(user_skills):
            score = cosine_similarity(
                job_embeddings[j_idx].reshape(1, -1),
                user_embeddings[u_idx].reshape(1, -1)
            )[0][0]
            scores.append(score)
            results.append((js, us, float(score)))

    def score_to_percent(score):
        score = max(0, min(score, 0.5))
        return (score / 0.5) * 100

    for js, us, score in results:
        pct = score_to_percent(score)
        print(f"{js} ↔ {us} → {pct:.2f}%")

    avg_score = np.mean(scores)
    raw_pct = score_to_percent(avg_score)

    if raw_pct > 95:
        excess = raw_pct - 95
        avg_pct = 95 + (excess / 10)
    else:
        avg_pct = raw_pct

    return raw_pct


#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Railway's dynamic port
    app.run(host="0.0.0.0", port=port)
