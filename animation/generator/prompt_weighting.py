# -*- coding: utf-8 -*-
"""This module exposes helper functions to provide prompt weighting"""

import numexpr
import re


def get_learned_conditioning(model, weighted_subprompts, text, args, sign=1):
    """Gets learned conditioning for prompt weighting"""

    if len(weighted_subprompts) < 1:
        log_tokenization(text, model, args.log_weighted_subprompts, sign)
        c = model.get_learned_conditioning(args.n_samples * [text])
    else:
        c = None
        for subtext, subweight in weighted_subprompts:
            log_tokenization(
                subtext, model, args.log_weighted_subprompts, sign * subweight
            )
            if c is None:
                c = model.get_learned_conditioning(args.n_samples * [subtext])
                c *= subweight
            else:
                c.add_(
                    model.get_learned_conditioning(args.n_samples * [subtext]),
                    alpha=subweight,
                )

    return c


def split_weighted_subprompts(text, frame=0, skip_normalize=False):
    """
    Grabs all text up to the first occurrence of ':'
    Uses the grabbed text as a sub-prompt, and takes the value following ':' as weight if ':' has no value defined,
    defaults to 1.0 repeats until no text remaining
    """

    prompt_parser = re.compile(
        """
            (?P<prompt>         # capture group for 'prompt'
            (?:\\\:|[^:])+      # match one or more non ':' characters or escaped colons '\:'
            )                   # end 'prompt'
            (?:                 # non-capture group
            :+                  # match one or more ':' characters
            (?P<weight>((        # capture group for 'weight'
            -?\d+(?:\.\d+)?     # match positive or negative integer or decimal number
            )|(                 # or
            `[\S\s]*?`# a math function
            )))?                 # end weight capture group, make optional
            \s*                 # strip spaces after weight
            |                   # OR
            $                   # else, if no ':' then match end of line
            )                   # end non-capture group
            """,
        re.VERBOSE,
    )

    negative_prompts = []
    positive_prompts = []

    for match in re.finditer(prompt_parser, text):
        w = parse_weight(match, frame)
        if w < 0:
            # negating the sign as we'll feed this to uc
            negative_prompts.append((match.group("prompt").replace("\\:", ":"), -w))
        elif w > 0:
            positive_prompts.append((match.group("prompt").replace("\\:", ":"), w))

    if skip_normalize:
        return (negative_prompts, positive_prompts)
    return (
        normalize_prompt_weights(negative_prompts),
        normalize_prompt_weights(positive_prompts),
    )


def parse_weight(match, frame=0):
    """Parse weight from match"""

    w_raw = match.group("weight")

    if w_raw == None:
        return 1

    if check_is_number(w_raw):
        return float(w_raw)
    else:
        t = frame
        # the value inside the characters cannot represent a math function
        if len(w_raw) < 3:
            return 1

        return float(numexpr.evaluate(w_raw[1:-1]))


def normalize_prompt_weights(parsed_prompts):
    """Normalize prompt weights between all prompts"""

    if len(parsed_prompts) == 0:
        return parsed_prompts

    weight_sum = sum(map(lambda x: x[1], parsed_prompts))

    # subprompt weights add up to zero, discard and use even weights instead
    if weight_sum == 0:
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]

    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]


def log_tokenization(text, model, log=False, weight=1):
    """Shows how the prompt is tokenized
    Usually tokens have '</w>' to indicate end-of-word, but for readability it has been replaced with ' '
    """

    if not log:
        return

    tokens = model.cond_stage_model.tokenizer._tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)

    for i in range(0, totalTokens):
        token = tokens[i].replace("</w>", " ")
        # alternate color
        s = (usedTokens % 6) + 1
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"

    print(f"\n>> Tokens ({usedTokens}), Weight ({weight:.2f}):\n{tokenized}\x1b[0m")

    if discarded != "":
        print(f">> Tokens Discarded ({totalTokens-usedTokens}):\n{discarded}\x1b[0m")


def check_is_number(value):
    """Check if input value is a number"""

    float_pattern = r"^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$"
    return re.match(float_pattern, value)
