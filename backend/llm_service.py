from config import GEMINI_API_KEY, ANTHROPIC_API_KEY, get_llm_provider

SYSTEM_PROMPT = """You are AI Law Advisor, a friendly and empathetic legal assistant for Indian law.
You explain legal rights in plain, conversational language — like a knowledgeable friend who happens to be a lawyer.

Style:
- Be warm, empathetic, and conversational — acknowledge the user's situation first before diving into law
- Use simple language (8th-grade level) but sound confident and reassuring
- Start with a brief empathetic acknowledgment (e.g., "That's a frustrating situation" or "I understand your concern")
- Use short paragraphs, not walls of text
- Bold key terms and amounts for easy scanning
- Use markdown formatting: headers, bullets, blockquotes for citations

Rules:
1. ALWAYS cite specific sections from the provided legal context using blockquotes
2. Give clear, numbered actionable steps the user can take RIGHT NOW
3. Mention timelines and deadlines where applicable (e.g., "you must file within 2 years")
4. Include relevant legal aid resources (helpline numbers, websites) when appropriate
5. NEVER provide legal advice — only legal information. End with a disclaimer
6. ALWAYS try to help. Even vague queries have a legal angle — find it
7. NEVER reject a query as "not legal"
8. For greetings, respond warmly and ask how you can help
9. If context is insufficient, still give general guidance and suggest where to get help

Respond in {language}."""

def _build_prompt(query, context_chunks, language='English'):
    context = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in context_chunks])
    return f"""{SYSTEM_PROMPT.format(language=language)}

## Legal Context:
{context}

## User Question:
{query}

Provide a clear, helpful response based on the legal context above."""


async def generate_response(query, context_chunks, language='en'):
    lang_map = {'en': 'English', 'hi': 'Hindi'}
    lang = lang_map.get(language, 'English')
    prompt = _build_prompt(query, context_chunks, lang)
    provider = get_llm_provider()

    if provider == 'gemini':
        return await _gemini_generate(prompt)
    elif provider == 'claude':
        return await _claude_generate(prompt)
    else:
        return _fallback_response(query, context_chunks)


async def _gemini_generate(prompt):
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text


async def _claude_generate(prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def _fallback_response(query, context_chunks):
    if not context_chunks:
        return "I couldn't find relevant legal information for your query. Please try rephrasing or check the Legal Aid page for professional assistance."

    sections = []
    for c in context_chunks[:3]:
        sections.append(f"**{c['source']}**: {c['text'][:200]}...")

    return f"""## Relevant Legal Information

Based on your question, here are the relevant legal provisions:

{chr(10).join('- ' + s for s in sections)}

> **Note:** No AI API key is configured. Add a `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` to your `.env` file for AI-powered plain language explanations.

*For personalized legal advice, please consult a qualified lawyer.*"""
