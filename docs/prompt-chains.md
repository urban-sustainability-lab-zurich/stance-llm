# Prompt Chains Documentation

This document outlines the prompt construction functions used in the stance-llm system for both German (`de`) and English (`en`). Each prompt construction function is shown with its template in both languages.

The different prompt chains (see the main Readme and the figures in [`docs/figures`](docs/figures)) combine these different functions in different control flows.

---

## 1. Irrelevance Prompt

**Function:** 

```
construct_irrelevance_prompt(input_text, entity, statement, language)
```

- **German (`de`):**
  ```
  Analysiere den folgenden Text: {input_text}. Bezieht die Organisation {entity} Stellung zur folgenden Aussage: {statement}? Beziehe dich nur auf den Text. Antworte mit Bezieht keine Stellung oder Bezieht Stellung
  ```

- **English (`en`):**
  ```
  Analyze the following text: {input_text}. Does the organization {entity} take a stance on the following statement: {statement}? Refer only to the text. Answer with Does not take a stance or Takes a stance
  ```

---

## 2. Summary Prompt

**Function:** 

```
construct_summary_prompt(input_text, entity, language)
```

- **German (`de`):**
  ```
  Fasse die Position der Organisation {entity} im folgenden Text zusammen: 
  {input_text}. Fasse dich kurz und starte deine Zusammenfassung mit: Die Organisation {entity}...
  ```

- **English (`en`):**
  ```
  Summarize the position of the organization {entity} in the following text:
  {input_text}. Be brief and start your summary with: The organization {entity}...
  ```

---

## 3. Statement-Specific Summary Prompt

**Function:** 

```
construct_summary_statementspecific_prompt(input_text, entity, statement, language)
```

- **German (`de`):**
  ```
  Fasse die Position der Organisation {entity} im folgenden Text in Bezug auf die Aussage "{statement}" zusammen: 
  {input_text}. 
  Fasse dich kurz und starte deine Zusammenfassung mit: Die Organisation {entity}...
  ```

- **English (`en`):**
  ```
  Summarize the position of the organization {entity} in the following text with respect to the statement "{statement}":
  {input_text}.
  Be brief and start your summary with: The organization {entity}...
  ```

---

## 4. General Stance Prompt

**Function:**

```
construct_general_stance_prompt(input_text, entity, language)
```

- **German (`de`):**
  ```
  Analysiere den folgenden Text: {input_text}. Bezieht die Organisation {entity} implizit oder explizit Stellung zu einem Sachverhalt? Beziehe dich nur auf den Text. Antworte mit Bezieht Stellung oder Bezieht keine Stellung
  ```

- **English (`en`):**
  ```
  Analyze the following text: {input_text}. Does the organization {entity} express an implicit or explicit position regarding an issue? Refer only to the text. Answer with Does not express a position or Expresses a position
  ```

---

## 5. Support Stance Prompt

**Function:** 

```
construct_support_stance_prompt(input_text, entity, statement, language)
```

- **German (`de`):**
  ```
  Analysiere den folgenden Text: {input_text}. Befürwortet die Organisation {entity} die Aussage: {statement}? Beziehe dich nur auf den Text. Antworte mit Ja oder Nein
  ```

- **English (`en`):**
  ```
  Analyze the following text: {input_text}. Does the organization {entity} support the statement: {statement}? Refer only to the text. Answer with Yes or No
  ```

---

## 6. Opposition Stance Prompt

**Function:** 

```
construct_opposition_stance_prompt(input_text, entity, statement, language)
```

- **German (`de`):**
  ```
  Analysiere den folgenden Text: {input_text}. Lehnt die Organisation {entity} die folgende Aussage ab: {statement}? Beziehe dich nur auf den Text. Antworte mit Ja oder Nein
  ```

- **English (`en`):**
  ```
  Analyze the following text: {input_text}. Does the organization {entity} oppose the statement: {statement}? Refer only to the text. Answer with Yes or No
  ```

---

## 7. Stance categorization based on constrained generation

Some chains prompt models to generate sentences starting with a fixed set of possible expressions. These are then used to categorize stances (see 8.):

- **German (`de`):**
  - `"drückt keine Haltung aus dazu, dass"` (does not express a position that)
  - `"unterstützt, dass"` (supports that)
  - `"lehnt ab, dass"` (opposes that)

- **English (`en`):**
  - `"does not express a position that"`
  - `"supports that"`
  - `"opposes that"`

---

## 8. Answer Mappings

### Irrelevance Answers

- **German (`de`):**
  - `"irrelevant"`: `Bezieht keine Stellung`
  - `"stance"`: `Bezieht Stellung`

- **English (`en`):**
  - `"irrelevant"`: `Does not take a stance`
  - `"stance"`: `Takes a stance`

### General Stance Answers (IRRELEVANCE_ANSWERS2)

- **German (`de`):**
  - `"irrelevant"`: `Bezieht Stellung`
  - `"stance"`: `Bezieht keine Stellung`

- **English (`en`):**
  - `"irrelevant"`: `Does not express a position`
  - `"stance"`: `Expresses a position`

---

## Usage

All prompt construction functions accept a `language` parameter (`"de"` or `"en"`). The default is `"de"` for backward compatibility.

Example:
```python
construct_irrelevance_prompt(text, entity, statement, language="en")
```

---