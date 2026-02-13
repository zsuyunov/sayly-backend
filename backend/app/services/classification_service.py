import os
import re
import requests
import logging
import time
from typing import Dict, Any, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Constants
HF_CLASSIFICATION_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ============================================================================
# STRICT CLASSIFICATION PROMPTS (used as candidate_labels for zero-shot)
# 
# CRITICAL INSTRUCTION: The model MUST use its semantic understanding and
# contextual knowledge FIRST. Keywords are only secondary helpers, not the
# primary classification method. The model should analyze meaning, intent,
# context, and semantic relationships - NOT just match keywords.
# ============================================================================

# Short label names mapped to the full labels (for clean output)
LABEL_SHORT_NAMES = {
    0: "gossip",
    1: "insult or unethical speech",
    2: "wasteful talk",
    3: "productive or meaningful speech"
}

# ============================================================================
# COMPREHENSIVE KEYWORD DICTIONARIES (SECONDARY HELPERS ONLY)
# 
# IMPORTANT: These keywords are ONLY used as minimal secondary helpers.
# The AI model's semantic understanding and contextual knowledge is PRIMARY.
# Keywords provide tiny boosts (0.02-0.10 max) to slightly assist the model,
# but they should NEVER override the model's semantic judgment.
# 
# Each category has hundreds of keywords/phrases for reference, but the
# model is explicitly instructed to use semantic understanding first.
# ============================================================================

# ---------- PRODUCTIVE / MEANINGFUL SPEECH KEYWORDS ----------
PRODUCTIVE_KEYWORDS = {
    # ---- Programming & Software Development ----
    "code", "coding", "coder", "programmer", "programming", "software", "hardware",
    "development", "developer", "devops", "frontend", "backend", "fullstack", "full stack",
    "algorithm", "algorithms", "data structure", "function", "functions", "variable", "variables",
    "debug", "debugging", "debugger", "error handling", "exception", "try catch",
    "compile", "compiler", "compilation", "runtime", "interpreter", "syntax",
    "loop", "iteration", "recursion", "recursive", "conditional", "boolean",
    "string", "integer", "float", "array", "list", "dictionary", "tuple", "set",
    "stack", "queue", "linked list", "tree", "graph", "hash map", "hash table",
    "binary search", "sorting", "merge sort", "quick sort", "bubble sort",
    "object oriented", "oop", "inheritance", "polymorphism", "encapsulation", "abstraction",
    "design pattern", "singleton", "factory", "observer", "mvc", "mvvm",
    "api", "rest api", "graphql", "endpoint", "request", "response", "http", "https",
    "server", "client", "database", "sql", "nosql", "mongodb", "postgresql", "mysql",
    "firebase", "firestore", "cloud", "aws", "azure", "gcp", "docker", "kubernetes",
    "microservice", "microservices", "architecture", "scalable", "scalability",
    "git", "github", "gitlab", "bitbucket", "commit", "branch", "merge", "pull request",
    "repository", "repo", "version control", "ci cd", "pipeline", "deployment", "deploy",
    "html", "css", "javascript", "typescript", "python", "java", "kotlin", "swift",
    "react", "angular", "vue", "svelte", "next.js", "nuxt", "gatsby",
    "node", "express", "django", "flask", "fastapi", "spring", "laravel",
    "flutter", "react native", "expo", "android", "ios", "mobile app",
    "machine learning", "deep learning", "neural network", "ai", "artificial intelligence",
    "natural language processing", "nlp", "computer vision", "tensorflow", "pytorch",
    "model", "training", "dataset", "feature engineering", "classification", "regression",
    "framework", "library", "module", "package", "dependency", "npm", "pip", "yarn",
    "import", "export", "class", "method", "constructor", "interface", "abstract",
    "async", "await", "promise", "callback", "event loop", "concurrency", "threading",
    "cursor", "editor", "ide", "vscode", "visual studio", "intellij", "pycharm",
    "terminal", "command line", "cli", "bash", "shell", "powershell",
    "script", "scripting", "automation", "cron job", "scheduler",
    "file", "directory", "path", "folder", "filesystem",
    "json", "xml", "yaml", "csv", "markdown", "regex", "regular expression",
    "unit test", "integration test", "test driven", "tdd", "jest", "pytest", "mocha",
    "documentation", "docs", "readme", "tutorial", "guide", "example", "sample",
    "refactor", "refactoring", "clean code", "code review", "pull request",
    "agile", "scrum", "sprint", "kanban", "jira", "trello", "backlog",
    "authentication", "authorization", "oauth", "jwt", "token", "session",
    "encryption", "security", "vulnerability", "penetration testing",
    "responsive", "accessibility", "seo", "performance", "optimization", "caching",
    "websocket", "socket", "real time", "streaming", "webhook",
    "containerization", "virtualization", "load balancer", "proxy", "nginx", "apache",
    "log", "logging", "monitoring", "analytics", "metrics", "dashboard",
    "linux", "ubuntu", "windows", "macos", "operating system",
    "network", "networking", "tcp", "udp", "dns", "ip address", "port",
    "bitmap", "pixel", "rendering", "gpu", "cpu", "memory", "ram",
    "blockchain", "cryptocurrency", "smart contract", "web3",
    "iot", "embedded", "raspberry pi", "arduino", "sensor",
    "open source", "license", "mit", "apache license",

    # ---- Education & Learning ----
    "learn", "learned", "learning", "learner", "lesson", "lessons",
    "teach", "teacher", "teaching", "taught", "instructor", "professor",
    "study", "studying", "student", "students", "academic", "academia",
    "education", "educational", "educate", "curriculum", "syllabus",
    "school", "university", "college", "institute", "classroom",
    "course", "courses", "class", "classes", "lecture", "lectures", "seminar",
    "exam", "examination", "quiz", "test", "assessment", "evaluation", "grading",
    "homework", "assignment", "project", "thesis", "dissertation", "research",
    "knowledge", "skill", "skills", "competency", "expertise", "mastery",
    "practice", "practicing", "exercise", "exercises", "drill", "training",
    "textbook", "book", "reading", "read", "chapter", "article", "paper",
    "workshop", "bootcamp", "certification", "certificate", "diploma", "degree",
    "scholarship", "fellowship", "internship", "apprenticeship",
    "mathematics", "math", "calculus", "algebra", "geometry", "statistics",
    "physics", "chemistry", "biology", "science", "scientific",
    "history", "geography", "economics", "psychology", "sociology", "philosophy",
    "literature", "language", "grammar", "vocabulary", "writing", "essay",
    "presentation", "slides", "powerpoint", "keynote",
    "understand", "understanding", "comprehend", "comprehension",
    "explain", "explaining", "explanation", "concept", "concepts", "theory",
    "formula", "equation", "principle", "principles", "law", "theorem",
    "experiment", "hypothesis", "observation", "analysis", "conclusion",
    "critical thinking", "problem solving", "logical", "reasoning",

    # ---- Professional Work & Career ----
    "work", "working", "worker", "workplace", "workforce",
    "job", "career", "profession", "professional", "occupation",
    "office", "company", "organization", "corporation", "enterprise",
    "business", "startup", "entrepreneur", "entrepreneurship",
    "manager", "management", "leadership", "leader", "supervisor",
    "team", "teamwork", "collaboration", "collaborate", "cooperate",
    "meeting", "meetings", "conference", "conference call", "video call",
    "agenda", "minutes", "action items", "follow up",
    "project", "project management", "milestone", "deadline", "deliverable",
    "task", "tasks", "assignment", "responsibility", "accountability",
    "goal", "goals", "objective", "objectives", "target", "kpi",
    "strategy", "strategic", "plan", "planning", "roadmap",
    "budget", "budgeting", "finance", "financial", "revenue", "profit",
    "marketing", "sales", "customer", "client", "stakeholder",
    "report", "reporting", "analysis", "analytics", "data",
    "process", "procedure", "workflow", "efficiency", "productivity",
    "innovation", "innovate", "creative", "creativity", "brainstorm",
    "proposal", "pitch", "negotiation", "contract", "agreement",
    "performance", "review", "feedback", "improvement", "growth",
    "hire", "hiring", "recruit", "recruitment", "interview",
    "training", "onboarding", "mentoring", "mentor", "coaching",
    "industry", "market", "competition", "competitive", "benchmark",
    "quality", "standard", "compliance", "regulation", "policy",
    "communication", "email", "memo", "announcement",
    "decision", "decide", "evaluate", "assessment", "criteria",
    "implement", "implementation", "execute", "execution", "rollout",
    "schedule", "calendar", "timeline", "prioritize", "priority",
    "resource", "resources", "allocation", "capacity",
    "risk", "risk management", "mitigation", "contingency",
    "supply chain", "logistics", "operations", "inventory",
    "profit", "loss", "roi", "return on investment", "margin",

    # ---- Health & Wellness (productive discussion) ----
    "health", "healthy", "wellness", "fitness", "exercise",
    "nutrition", "diet", "calories", "protein", "vitamins",
    "medical", "medicine", "doctor", "hospital", "treatment",
    "mental health", "therapy", "counseling", "wellbeing",
    "meditation", "mindfulness", "stress management", "self care",
    "workout", "gym", "running", "yoga", "sports",

    # ---- Science & Research ----
    "research", "researcher", "scientist", "discovery", "findings",
    "experiment", "laboratory", "lab", "sample", "specimen",
    "data", "dataset", "variable", "control group", "hypothesis",
    "methodology", "method", "approach", "framework", "model",
    "publication", "journal", "peer review", "citation",
    "innovation", "breakthrough", "patent", "invention",
    "climate", "environment", "sustainability", "renewable", "ecology",
    "vaccine", "virus", "bacteria", "genome", "dna", "rna",
    "astronomy", "space", "rocket", "satellite", "nasa",
    "engineering", "mechanical", "electrical", "civil", "chemical",

    # ---- Financial Literacy (productive) ----
    "investment", "investing", "stocks", "bonds", "portfolio",
    "savings", "saving", "budget", "financial planning", "retirement",
    "compound interest", "dividend", "mutual fund", "etf", "index fund",
    "mortgage", "insurance", "tax", "taxes", "deduction",
    "asset", "liability", "net worth", "cash flow", "income",
    "expense", "expenses", "accounting", "bookkeeping", "audit",

    # ---- Constructive / Goal-oriented Speech ----
    "improve", "improvement", "better", "progress", "advancing",
    "achieve", "achievement", "accomplish", "accomplishment", "milestone",
    "solve", "solution", "resolve", "resolution", "fix",
    "build", "building", "create", "creating", "develop", "developing",
    "design", "designing", "architect", "blueprint", "prototype",
    "optimize", "optimization", "efficient", "efficiency", "streamline",
    "contribute", "contribution", "volunteer", "volunteering",
    "impact", "meaningful", "purposeful", "intentional", "focused",
    "responsible", "accountability", "commitment", "dedicated", "dedication",
    "grateful", "gratitude", "thankful", "appreciate", "appreciation",
    "motivate", "motivation", "inspire", "inspiration", "encourage",
    "support", "helping", "assist", "guidance", "advise", "advice",
    "cooperate", "cooperation", "partnership", "alliance",
    "respect", "respectful", "dignity", "integrity", "honesty",
    "empathy", "compassion", "kindness", "generous", "generosity",
    "constructive", "positive", "productive", "meaningful",
    "reflect", "reflection", "self improvement", "personal growth",
    "discipline", "disciplined", "consistency", "persistent", "perseverance",
    "spiritual", "prayer", "worship", "quran", "bible", "scripture",
    "faith", "belief", "religious", "charity", "zakat", "sadaqah",
    "halal", "ethical", "moral", "morality", "virtue", "righteous",
}

# ---------- GOSSIP KEYWORDS ----------
GOSSIP_KEYWORDS = {
    # ---- Core gossip phrases ----
    "did you hear", "have you heard", "i heard that", "heard about",
    "did you know about", "did you know that", "guess what",
    "you won't believe", "you know what happened", "can you believe",
    "apparently", "supposedly", "allegedly", "i was told",
    "someone told me", "they told me", "she told me", "he told me",
    "people are saying", "people say", "everyone is saying", "everyone knows",
    "word on the street", "the word is", "word is",
    "between you and me", "between us", "just between us",
    "don't tell anyone", "keep this between us", "keep it secret",
    "don't spread this", "promise not to tell", "off the record",
    "i shouldn't be telling you", "i shouldn't say this",

    # ---- Talking about others' personal lives ----
    "she's dating", "he's dating", "they're dating", "they broke up",
    "she broke up", "he broke up", "they got divorced", "she got divorced",
    "he cheated", "she cheated", "they cheated", "having an affair",
    "secret relationship", "seeing someone", "hooking up",
    "she's pregnant", "he got her pregnant", "shotgun wedding",
    "family problems", "family drama", "their marriage", "their relationship",
    "his wife", "her husband", "his girlfriend", "her boyfriend",
    "what she did", "what he did", "what they did",
    "behind his back", "behind her back", "behind their back",

    # ---- Rumors and speculation ----
    "rumor", "rumors", "rumour", "rumours", "rumouring",
    "spreading rumors", "heard a rumor", "there's a rumor",
    "i think she", "i think he", "i bet she", "i bet he",
    "she probably", "he probably", "they probably",
    "she must be", "he must be", "they must be",
    "i wonder if she", "i wonder if he",

    # ---- Talking behind someone's back ----
    "talking behind", "behind someone's back", "when she's not around",
    "when he's not around", "when they're not here",
    "don't tell her", "don't tell him", "don't tell them",
    "she doesn't know", "he doesn't know", "they don't know",
    "if she finds out", "if he finds out",

    # ---- Judging others ----
    "she's so", "he's so", "they're so", "she's always", "he's always",
    "she never", "he never", "they never",
    "what's wrong with her", "what's wrong with him",
    "she thinks she's", "he thinks he's",
    "who does she think", "who does he think",
    "she's not as", "he's not as",
    "can you imagine her", "can you imagine him",

    # ---- Social drama and drama language ----
    "drama", "scandal", "scandalous", "controversial",
    "caught", "exposed", "busted", "found out",
    "spill the tea", "tea", "juicy", "juicy gossip",
    "dirt", "dirty secret", "dirty laundry",
    "backstabbing", "backstabber", "two faced", "two-faced",
    "fake", "fake friend", "fake person", "pretending",
    "jealous", "jealousy", "envious", "envy",
    "showing off", "bragging", "boasting", "flaunting",

    # ---- Workplace gossip ----
    "got fired", "getting fired", "about to be fired",
    "sleeping with", "sleeping around",
    "favoritism", "playing favorites", "teacher's pet", "boss's favorite",
    "promotion drama", "office politics", "office gossip",
    "talking about coworker", "talking about colleague",
    "behind the boss", "the boss doesn't know",
    "who's getting promoted", "who got the raise",
    "salary gossip", "how much they make", "how much she makes",

    # ---- Celebrity / public figure gossip ----
    "celebrity", "celebrities", "famous", "paparazzi",
    "tabloid", "entertainment news", "reality tv",
    "influencer drama", "youtuber drama", "twitter drama",
    "social media drama", "instagram drama", "tiktok drama",

    # ---- Reporting what others said ----
    "she was like", "he was like", "they were like",
    "she goes", "he goes", "and then she said",
    "and then he said", "and she was all",
    "you should have seen", "you should have heard",
    "i overheard", "eavesdropping", "eavesdrop",
    "whisper", "whispering", "hushed",
}

# ---------- INSULT / UNETHICAL SPEECH KEYWORDS ----------
UNETHICAL_KEYWORDS = {
    # ---- Direct insults ----
    "stupid", "idiot", "moron", "imbecile", "fool", "foolish",
    "dumb", "dumbass", "retard", "retarded",
    "loser", "pathetic", "worthless", "useless", "good for nothing",
    "ugly", "fat", "skinny", "disgusting", "repulsive", "hideous",
    "freak", "weirdo", "creep", "creepy", "psycho",
    "jerk", "scumbag", "lowlife", "trash", "garbage", "piece of",
    "shut up", "shut your mouth", "shut your face",
    "get lost", "go away", "nobody asked you", "nobody cares",
    "you're nothing", "you're nobody", "you don't matter",

    # ---- Profanity and vulgar language ----
    "damn", "hell", "crap", "crap",
    "wtf", "stfu", "lmao", "omfg",
    "ass", "asshole", "bastard",
    "suck", "sucks", "sucking",

    # ---- Threats and violence ----
    "kill", "murder", "die", "death", "dead",
    "beat up", "punch", "slap", "hit", "hurt",
    "destroy", "crush", "annihilate", "obliterate",
    "threat", "threaten", "threatening", "intimidate", "intimidation",
    "bully", "bullying", "harass", "harassment", "stalk", "stalking",
    "revenge", "payback", "get back at", "retaliate",
    "suffer", "suffering", "torture", "torment",
    "weapon", "gun", "knife", "bomb",

    # ---- Hate speech and discrimination ----
    "hate", "hatred", "despise", "detest", "loathe", "abhor",
    "racist", "racism", "racial slur", "racial profiling",
    "sexist", "sexism", "misogyny", "misogynist",
    "homophobic", "homophobia", "transphobic", "transphobia",
    "xenophobic", "xenophobia", "bigot", "bigotry", "bigoted",
    "discrimination", "discriminate", "discriminating",
    "prejudice", "prejudiced", "biased", "bias",
    "inferior", "superior", "supremacist", "supremacy",
    "stereotype", "stereotyping", "generalize",

    # ---- Manipulation and deception ----
    "liar", "lying", "lies", "deceive", "deceit", "deceitful",
    "cheat", "cheater", "cheating", "fraud", "fraudulent",
    "manipulate", "manipulation", "manipulative", "manipulator",
    "exploit", "exploitation", "exploiting", "take advantage",
    "betray", "betrayal", "traitor", "backstab",
    "corrupt", "corruption", "bribe", "bribery",
    "scam", "scammer", "con", "con artist", "swindle",
    "steal", "stealing", "thief", "theft", "rob", "robbery",
    "blackmail", "extort", "extortion",
    "gaslighting", "gaslight", "toxic", "toxicity",
    "abuse", "abusive", "abuser", "mistreat", "maltreat",

    # ---- Disrespect and contempt ----
    "disrespect", "disrespectful", "rude", "rudeness",
    "arrogant", "arrogance", "condescending", "patronizing",
    "mocking", "mock", "ridicule", "belittle", "demean", "demeaning",
    "humiliate", "humiliation", "shame", "shaming",
    "insult", "insulting", "offend", "offensive", "offense",
    "curse", "cursing", "swear", "swearing", "profanity",
    "vulgar", "vulgarity", "obscene", "obscenity",
    "contempt", "contemptuous", "scorn", "scornful", "disdain",
    "spite", "spiteful", "malice", "malicious", "malevolent",
    "cruel", "cruelty", "sadistic", "heartless", "ruthless",
    "wicked", "evil", "vile", "vicious", "nasty",

    # ---- Unethical behavior ----
    "unethical", "immoral", "amoral", "wrong", "sinful",
    "illegal", "unlawful", "criminal", "crime",
    "plagiarism", "plagiarize", "copy", "counterfeit",
    "bribery", "nepotism", "favoritism", "cronyism",
    "embezzle", "embezzlement", "launder", "money laundering",
}

# ---------- WASTEFUL TALK KEYWORDS ----------
WASTEFUL_KEYWORDS = {
    # ---- Filler words and phrases ----
    "umm", "uhh", "uh", "hmm", "hmmm",
    "like", "you know", "i mean", "basically", "literally",
    "whatever", "anyways", "anyhoo", "anyhow",
    "blah blah", "blah", "yada yada", "etc etc",
    "so yeah", "yeah yeah", "ya know",

    # ---- Aimless conversation ----
    "nothing much", "not much", "same old", "same old same old",
    "just chilling", "just hanging", "just vibing", "nothing new",
    "bored", "boring", "so bored", "nothing to do",
    "killing time", "wasting time", "passing time",
    "i don't know", "idk", "no idea", "who knows", "who cares",
    "doesn't matter", "don't care", "whatever",

    # ---- Repetitive meaningless phrases ----
    "right right", "yeah yeah", "okay okay", "sure sure",
    "true true", "exactly exactly",
    "i guess", "i suppose", "maybe", "perhaps", "possibly",
    "sort of", "kind of", "kinda", "sorta",
    "thing", "thingy", "thingamajig", "stuff", "things",
    "whatever it is", "you know what i mean",

    # ---- Trivial topics (no purpose) ----
    "what's up", "sup", "how's it going", "how's it hanging",
    "hey", "hi", "hello", "yo", "howdy",
    "see ya", "later", "bye bye", "take care",
    "nice weather", "cold today", "hot today", "rain",
    "what time is it", "what day is it",
    "random thought", "just thinking", "just saying",
    "lol", "haha", "hehe", "rofl", "lmfao",
    "wait what", "hold on", "hang on", "one sec",
    "no way", "seriously", "for real", "oh really",
    "that's crazy", "that's wild", "that's insane",
    "oh my god", "oh my gosh", "omg",
}


def _build_classification_labels_with_keywords() -> List[str]:
    """
    Build classification labels that include explicit instructions and keyword lists.
    These are sent directly to the AI model as prompts.
    """
    # Format keywords as comma-separated lists (limit to reasonable length for API)
    gossip_keywords_str = ", ".join(list(GOSSIP_KEYWORDS)[:100])  # First 100 keywords
    unethical_keywords_str = ", ".join(list(UNETHICAL_KEYWORDS)[:100])
    wasteful_keywords_str = ", ".join(list(WASTEFUL_KEYWORDS)[:100])
    productive_keywords_str = ", ".join(list(PRODUCTIVE_KEYWORDS)[:150])  # More for productive
    
    return [
        # GOSSIP: Talking about absent people, their private lives, spreading rumors
        f"""CRITICAL INSTRUCTION FOR AI MODEL: You MUST use your semantic understanding and contextual knowledge FIRST as the PRIMARY classification method. Keywords are ONLY secondary helpers. Do NOT rely on keyword matching alone.

CLASSIFICATION TASK: Determine if this speech is gossip. This speech is gossip if it discusses other people behind their backs, spreads rumors, talks about someone's private life, personal affairs, relationships, or shares unverified information about absent individuals.

WORKFLOW:
1. PRIMARY STEP: Use your semantic understanding to analyze the full meaning, intent, and context of the speech. Ask yourself: Is the speaker talking about absent people's private matters? Is there malicious intent or spreading of unverified information? Analyze the overall context and meaning, not individual words.
2. SECONDARY STEP: Only AFTER semantic analysis, you may use these keywords as helpers: {gossip_keywords_str}

REMEMBER: Semantic understanding is PRIMARY. Keywords are SECONDARY helpers only. If the semantic meaning indicates gossip but keywords are absent, still classify as gossip. If keywords are present but semantic meaning is NOT gossip, do NOT classify as gossip.""",

        # INSULT / UNETHICAL: Offensive, harmful, discriminatory, abusive language
        f"""CRITICAL INSTRUCTION FOR AI MODEL: You MUST use your semantic understanding and contextual knowledge FIRST as the PRIMARY classification method. Keywords are ONLY secondary helpers. Do NOT rely on keyword matching alone.

CLASSIFICATION TASK: Determine if this speech contains insults, offensive language, verbal abuse, hate speech, threats, bullying, discrimination, profanity, cursing, or ethically harmful and toxic statements.

WORKFLOW:
1. PRIMARY STEP: Use your semantic understanding to analyze the full meaning, intent, tone, and context of the speech. Ask yourself: Does the speech have harmful intent? Is the tone offensive? Is the message discriminatory or abusive? Analyze the overall meaning and context, not individual words. Even if profanity is absent, if the semantic meaning is harmful, classify accordingly.
2. SECONDARY STEP: Only AFTER semantic analysis, you may use these keywords as helpers: {unethical_keywords_str}

REMEMBER: Semantic understanding is PRIMARY. Keywords are SECONDARY helpers only. If the semantic meaning indicates harmful/unethical content but keywords are absent, still classify as unethical. If keywords are present but semantic meaning is NOT harmful, do NOT classify as unethical.""",

        # WASTEFUL: Aimless, repetitive, no-purpose chatter
        f"""CRITICAL INSTRUCTION FOR AI MODEL: You MUST use your semantic understanding and contextual knowledge FIRST as the PRIMARY classification method. Keywords are ONLY secondary helpers. Do NOT rely on keyword matching alone.

CLASSIFICATION TASK: Determine if this speech is wasteful - idle small talk with no real purpose, aimless rambling, repetitive meaningless chatter, time-wasting conversation, or trivial banter about nothing important.

WORKFLOW:
1. PRIMARY STEP: Use your semantic understanding to analyze the full meaning, purpose, and value of the speech. Ask yourself: Does the speech have a clear purpose? Is there meaningful substance? Is it genuinely aimless or just casual conversation? Analyze the overall meaning and value, not individual words. Casual conversation with purpose is NOT wasteful.
2. SECONDARY STEP: Only AFTER semantic analysis, you may use these keywords as helpers: {wasteful_keywords_str}

REMEMBER: Semantic understanding is PRIMARY. Keywords are SECONDARY helpers only. If the semantic meaning indicates wasteful talk but keywords are absent, still classify as wasteful. If keywords are present but semantic meaning has purpose, do NOT classify as wasteful.""",

        # PRODUCTIVE: Educational, professional, constructive, goal-oriented
        f"""CRITICAL INSTRUCTION FOR AI MODEL: You MUST use your semantic understanding and contextual knowledge FIRST as the PRIMARY classification method. Keywords are ONLY secondary helpers. Do NOT rely on keyword matching alone.

CLASSIFICATION TASK: Determine if this speech is productive and meaningful - involving education, learning, teaching, professional work discussion, problem-solving, technical explanation, sharing knowledge, planning, constructive feedback, or goal-oriented dialogue.

WORKFLOW:
1. PRIMARY STEP: Use your semantic understanding to analyze the full meaning, value, and purpose of the speech. Ask yourself: Does the speech have educational value? Is it constructive? Does it involve problem-solving or knowledge sharing? Analyze the overall meaning and value, not individual words. Even without technical terms, if the semantic meaning is productive, classify accordingly.
2. SECONDARY STEP: Only AFTER semantic analysis, you may use these keywords as helpers: {productive_keywords_str}

REMEMBER: Semantic understanding is PRIMARY. Keywords are SECONDARY helpers only. If the semantic meaning indicates productive speech but keywords are absent, still classify as productive. If keywords are present but semantic meaning is NOT productive, do NOT classify as productive."""
    ]


# Build classification labels dynamically with keywords included
CLASSIFICATION_LABELS = _build_classification_labels_with_keywords()


class HuggingFaceClassificationService:
    def __init__(self):
        self.api_key = os.getenv("HF_API_KEY")
        if not self.api_key:
            logger.error("HF_API_KEY environment variable not set")
            raise ValueError("HF_API_KEY environment variable not set")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # Pre-compile keyword patterns for fast matching
        # Use word boundaries for single words, direct match for phrases
        self._productive_singles = set()
        self._productive_phrases = []
        for kw in PRODUCTIVE_KEYWORDS:
            if " " in kw:
                self._productive_phrases.append(kw)
            else:
                self._productive_singles.add(kw)

        self._gossip_singles = set()
        self._gossip_phrases = []
        for kw in GOSSIP_KEYWORDS:
            if " " in kw:
                self._gossip_phrases.append(kw)
            else:
                self._gossip_singles.add(kw)

        self._unethical_singles = set()
        self._unethical_phrases = []
        for kw in UNETHICAL_KEYWORDS:
            if " " in kw:
                self._unethical_phrases.append(kw)
            else:
                self._unethical_singles.add(kw)

        self._wasteful_singles = set()
        self._wasteful_phrases = []
        for kw in WASTEFUL_KEYWORDS:
            if " " in kw:
                self._wasteful_phrases.append(kw)
            else:
                self._wasteful_singles.add(kw)

        print(f"[CLASSIFICATION] Keyword dictionaries loaded: "
              f"productive={len(PRODUCTIVE_KEYWORDS)}, gossip={len(GOSSIP_KEYWORDS)}, "
              f"unethical={len(UNETHICAL_KEYWORDS)}, wasteful={len(WASTEFUL_KEYWORDS)}")

    # ------------------------------------------------------------------
    # Keyword detection engine
    # ------------------------------------------------------------------
    def _count_keyword_hits(self, text_lower: str, words_set: set) -> Tuple[int, List[str]]:
        """Count how many unique words from words_set appear in the text."""
        # Tokenise text into words
        text_words = set(re.findall(r"[a-z'\-]+", text_lower))
        hits = text_words & words_set
        return len(hits), list(hits)[:10]  # return up to 10 example hits

    def _count_phrase_hits(self, text_lower: str, phrases: List[str]) -> Tuple[int, List[str]]:
        """Count how many phrases appear in the text."""
        hits = []
        for phrase in phrases:
            if phrase in text_lower:
                hits.append(phrase)
        return len(hits), hits[:10]

    def _detect_keywords(self, text: str) -> Dict[str, Any]:
        """
        Run comprehensive keyword analysis on text.
        
        IMPORTANT: This is ONLY for generating minimal secondary helper boosts.
        The AI model's semantic understanding is PRIMARY. These keywords provide
        tiny nudges (0.02-0.10 max) to slightly assist the model, NOT to override
        its semantic judgment.

        Returns dict with:
          - counts: {category: int} - keyword hit counts (for reference)
          - boosts: {category: float} - minimal boost values (0.02-0.10 max)
          - matched: {category: [str]} - sample of matched keywords (for logging)
          - densities: {category: float} - keyword density (for reference)
        """
        text_lower = text.lower()
        total_words = max(len(text_lower.split()), 1)

        results: Dict[str, Any] = {
            "counts": {},
            "boosts": {},
            "matched": {},
            "densities": {},
        }

        # --- Productive ---
        # NOTE: Keywords are ONLY small helpers. The AI model's semantic understanding
        # is PRIMARY. These boosts are minimal nudges (max 0.08) to slightly assist
        # the model, not override its judgment.
        w_hits, w_ex = self._count_keyword_hits(text_lower, self._productive_singles)
        p_hits, p_ex = self._count_phrase_hits(text_lower, self._productive_phrases)
        prod_total = w_hits + p_hits
        prod_density = prod_total / total_words
        results["counts"]["productive"] = prod_total
        results["matched"]["productive"] = (w_ex + p_ex)[:10]
        results["densities"]["productive"] = round(prod_density, 4)
        # Small boost only - model's semantic understanding is primary
        if prod_total >= 8:
            results["boosts"]["productive"] = 0.08
        elif prod_total >= 5:
            results["boosts"]["productive"] = 0.06
        elif prod_total >= 3:
            results["boosts"]["productive"] = 0.04
        elif prod_total >= 1:
            results["boosts"]["productive"] = 0.02
        else:
            results["boosts"]["productive"] = 0.0

        # --- Gossip ---
        # NOTE: Keywords are ONLY small helpers. The AI model's semantic understanding
        # is PRIMARY. These boosts are minimal nudges (max 0.08) to slightly assist
        # the model, not override its judgment.
        w_hits, w_ex = self._count_keyword_hits(text_lower, self._gossip_singles)
        p_hits, p_ex = self._count_phrase_hits(text_lower, self._gossip_phrases)
        gossip_total = w_hits + p_hits
        gossip_density = gossip_total / total_words
        results["counts"]["gossip"] = gossip_total
        results["matched"]["gossip"] = (w_ex + p_ex)[:10]
        results["densities"]["gossip"] = round(gossip_density, 4)
        # Small boost only - model's semantic understanding is primary
        if gossip_total >= 5:
            results["boosts"]["gossip"] = 0.08
        elif gossip_total >= 3:
            results["boosts"]["gossip"] = 0.06
        elif gossip_total >= 2:
            results["boosts"]["gossip"] = 0.04
        elif gossip_total >= 1:
            results["boosts"]["gossip"] = 0.02
        else:
            results["boosts"]["gossip"] = 0.0

        # --- Unethical ---
        # NOTE: Keywords are ONLY small helpers. The AI model's semantic understanding
        # is PRIMARY. These boosts are minimal nudges (max 0.10) to slightly assist
        # the model, not override its judgment. Unethical gets slightly higher boost
        # due to critical importance, but still minimal.
        w_hits, w_ex = self._count_keyword_hits(text_lower, self._unethical_singles)
        p_hits, p_ex = self._count_phrase_hits(text_lower, self._unethical_phrases)
        unethical_total = w_hits + p_hits
        unethical_density = unethical_total / total_words
        results["counts"]["unethical"] = unethical_total
        results["matched"]["unethical"] = (w_ex + p_ex)[:10]
        results["densities"]["unethical"] = round(unethical_density, 4)
        # Small boost only - model's semantic understanding is primary
        if unethical_total >= 4:
            results["boosts"]["unethical"] = 0.10
        elif unethical_total >= 2:
            results["boosts"]["unethical"] = 0.07
        elif unethical_total >= 1:
            results["boosts"]["unethical"] = 0.04
        else:
            results["boosts"]["unethical"] = 0.0

        # --- Wasteful ---
        # NOTE: Keywords are ONLY small helpers. The AI model's semantic understanding
        # is PRIMARY. These boosts are minimal nudges (max 0.08) to slightly assist
        # the model, not override its judgment.
        w_hits, w_ex = self._count_keyword_hits(text_lower, self._wasteful_singles)
        p_hits, p_ex = self._count_phrase_hits(text_lower, self._wasteful_phrases)
        wasteful_total = w_hits + p_hits
        wasteful_density = wasteful_total / total_words
        results["counts"]["wasteful"] = wasteful_total
        results["matched"]["wasteful"] = (w_ex + p_ex)[:10]
        results["densities"]["wasteful"] = round(wasteful_density, 4)
        # Small boost only - model's semantic understanding is primary
        if wasteful_total >= 6:
            results["boosts"]["wasteful"] = 0.08
        elif wasteful_total >= 4:
            results["boosts"]["wasteful"] = 0.06
        elif wasteful_total >= 2:
            results["boosts"]["wasteful"] = 0.04
        else:
            results["boosts"]["wasteful"] = 0.0

        return results

    # ------------------------------------------------------------------
    # Main classification method
    # ------------------------------------------------------------------
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text using Hugging Face Zero-Shot Classification.
        
        IMPORTANT: The AI model uses its semantic understanding and contextual
        knowledge as the PRIMARY classification method. Keywords are only used
        as small secondary helpers (minimal boosts) to slightly assist the model,
        NOT to override its judgment.
        """
        if not text or not text.strip():
            print("[CLASSIFICATION] Empty text provided, skipping classification")
            return {"labels": [], "scores": []}

        text_length = len(text)
        text_preview = text[:120] + "..." if len(text) > 120 else text
        print(f"[CLASSIFICATION] Starting classification (text length: {text_length} chars)")
        print(f"[CLASSIFICATION] Text preview: {text_preview}")
        print(f"[CLASSIFICATION] PRIMARY: Using AI model's semantic understanding and contextual knowledge")
        print(f"[CLASSIFICATION] SECONDARY: Keywords will provide minimal boosts only (max 0.08-0.10)")

        # ---- Keyword analysis (SECONDARY - only for small boosts) ----
        kw = self._detect_keywords(text)
        print(f"[CLASSIFICATION] Keyword hits (for reference only): productive={kw['counts']['productive']}, "
              f"gossip={kw['counts']['gossip']}, unethical={kw['counts']['unethical']}, "
              f"wasteful={kw['counts']['wasteful']}")
        if kw["matched"]["productive"]:
            print(f"[CLASSIFICATION] Productive keywords matched (helper only): {kw['matched']['productive']}")
        if kw["matched"]["gossip"]:
            print(f"[CLASSIFICATION] Gossip keywords matched (helper only): {kw['matched']['gossip']}")
        if kw["matched"]["unethical"]:
            print(f"[CLASSIFICATION] Unethical keywords matched (helper only): {kw['matched']['unethical']}")
        if kw["matched"]["wasteful"]:
            print(f"[CLASSIFICATION] Wasteful keywords matched (helper only): {kw['matched']['wasteful']}")

        boosts = kw["boosts"]

        # ---- Build API payload ----
        # The model will use its semantic understanding FIRST based on the strict prompts
        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": CLASSIFICATION_LABELS,
                "multi_label": False,
            }
        }

        # ---- Retry loop ----
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if attempt > 1:
                    wait_time = RETRY_DELAY * (attempt - 1)
                    print(f"[CLASSIFICATION] Retry attempt {attempt}/{MAX_RETRIES} after {wait_time}s delay")
                    time.sleep(wait_time)

                print(f"[CLASSIFICATION] Sending request to Hugging Face API (attempt {attempt}/{MAX_RETRIES})")
                response = requests.post(
                    HF_CLASSIFICATION_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code != 200:
                    error_text = response.text
                    if len(error_text) > 500 or "<!DOCTYPE" in error_text:
                        if "504" in error_text:
                            error_text = "504 Gateway Timeout"
                        elif "503" in error_text:
                            error_text = "503 Service Unavailable"
                        elif "502" in error_text:
                            error_text = "502 Bad Gateway"
                        else:
                            error_text = error_text[:200] + "..."
                    print(f"[CLASSIFICATION] API error {response.status_code}: {error_text}")
                    if response.status_code in [502, 503, 504, 429] and attempt < MAX_RETRIES:
                        last_error = Exception(f"HF API error {response.status_code}: {error_text}")
                        continue
                    raise Exception(f"HF API error {response.status_code}: {error_text}")

                result = response.json()

                # ---- Normalise response shape ----
                if isinstance(result, list):
                    if len(result) == 0:
                        raise Exception("Empty list response from Hugging Face API")
                    if isinstance(result[0], dict) and "label" in result[0]:
                        labels = [item["label"] for item in result]
                        scores = [item["score"] for item in result]
                        result = {"labels": labels, "scores": scores}
                    else:
                        result = result[0]

                if not isinstance(result, dict):
                    raise Exception(f"Unexpected response type: {type(result)}")

                if "label" in result and "labels" not in result:
                    result = {"labels": [result["label"]], "scores": [result["score"]]}

                if "labels" not in result or "scores" not in result:
                    raise Exception(f"Missing labels/scores in response: {list(result.keys())}")

                labels = result["labels"]
                scores = list(result["scores"])  # make mutable copy

                # ---- Map full labels to indices ----
                cat_index = {}  # category_key -> index
                for i, label in enumerate(labels):
                    label_lower = label.lower()
                    if "gossip" in label_lower:
                        cat_index["gossip"] = i
                    elif "insult" in label_lower or "unethical" in label_lower:
                        cat_index["unethical"] = i
                    elif "wasteful" in label_lower or "idle" in label_lower or "aimless" in label_lower:
                        cat_index["wasteful"] = i
                    elif "productive" in label_lower or "meaningful" in label_lower:
                        cat_index["productive"] = i

                # ---- Apply minimal keyword boosts (SECONDARY HELPERS ONLY) ----
                # IMPORTANT: These are tiny nudges (0.02-0.10 max) to slightly assist
                # the model. The model's semantic understanding is PRIMARY and these
                # boosts should never override the model's judgment.
                print(f"[CLASSIFICATION] AI model's semantic classification complete. Applying minimal keyword boosts (helpers only)...")
                for cat_key, boost in boosts.items():
                    if boost > 0 and cat_key in cat_index:
                        idx = cat_index[cat_key]
                        old_score = scores[idx]
                        scores[idx] = min(1.0, scores[idx] + boost)
                        print(f"[CLASSIFICATION] Applied minimal boost to {cat_key}: {old_score:.3f} -> {scores[idx]:.3f} (+{boost:.3f} helper boost)")

                # ---- Re-sort by score descending ----
                pairs = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
                labels = [l for l, _ in pairs]
                scores = [s for _, s in pairs]
                result = {"labels": labels, "scores": scores}

                # ---- Log results ----
                clean = [LABEL_SHORT_NAMES.get(i, labels[i].split(":")[0].strip())
                         for i in range(len(labels))]
                # Rebuild clean list based on actual label content
                clean_labels = []
                for label in labels:
                    ll = label.lower()
                    if "gossip" in ll:
                        clean_labels.append("gossip")
                    elif "insult" in ll or "unethical" in ll:
                        clean_labels.append("insult or unethical speech")
                    elif "wasteful" in ll or "idle" in ll or "aimless" in ll:
                        clean_labels.append("wasteful talk")
                    elif "productive" in ll or "meaningful" in ll:
                        clean_labels.append("productive or meaningful speech")
                    else:
                        clean_labels.append(label[:40])

                top_clean = clean_labels[0] if clean_labels else "unknown"
                print(f"[CLASSIFICATION] Classification successful!")
                print(f"[CLASSIFICATION] Top category: {top_clean} (confidence: {scores[0]:.3f})")
                print(f"[CLASSIFICATION] All scores: {dict(zip(clean_labels, [f'{s:.3f}' for s in scores]))}")

                return result

            except requests.exceptions.Timeout:
                print("[CLASSIFICATION] Request timeout after 30s")
                if attempt < MAX_RETRIES:
                    last_error = Exception("Request timeout after 30s")
                    continue
                raise Exception("Request timeout after 30s")
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                if hasattr(e, "response") and e.response is not None:
                    status_code = e.response.status_code
                    if status_code in [502, 503, 504, 429] and attempt < MAX_RETRIES:
                        last_error = Exception(f"Classification failed: {error_msg}")
                        continue
                print(f"[CLASSIFICATION] Request failed: {error_msg[:200]}")
                raise Exception(f"Classification failed: {error_msg}")
            except Exception as e:
                error_msg = str(e)
                if "HF API error" in error_msg and any(c in error_msg for c in ["502", "503", "504", "429"]) and attempt < MAX_RETRIES:
                    last_error = e
                    continue
                print(f"[CLASSIFICATION] Classification error: {error_msg[:200]}")
                raise

        if last_error:
            print(f"[CLASSIFICATION] All {MAX_RETRIES} retry attempts failed")
            raise last_error


# ============================================================================
# Public API
# ============================================================================
_service = None


def classify_speech_text(text: str) -> Dict[str, Any]:
    """Wrapper function for classification."""
    global _service
    if not _service:
        print("[CLASSIFICATION] Initializing Hugging Face Classification Service")
        _service = HuggingFaceClassificationService()
    return _service.classify_text(text)
