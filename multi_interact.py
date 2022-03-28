import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import googletrans

import torch
import torch.nn.functional as F

from config import InteractConfig
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, GPT2LMHeadModel
from utils import download_pretrained_model, get_dataset


def build_input_from_segments(topic, history, emotions, actions,reply, tokenizer, SPECIAL_TOKENS, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:4])
    no_emo, happy = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[4:6])
    direct, inform = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[20:22])
    instance = {}
    sequence = [[bos] + topic] + history + [reply + ([eos] if with_eos else [])]  
    sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                enumerate(sequence[1:])]
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]  # the last for is for repeating the speaker1 and speaker2 for all tokens
    emo = [[no_emo] + list(chain(*[emotions[i] for i, s in enumerate(sequence[1:-1]) for _ in s])) + [happy] * len(sequence[-1])]
    acts = [[inform] + list(chain(*[actions[i] for i, s in enumerate(sequence[1:-1])for _ in s])) + [inform] * len(sequence[-1])]
    instance["token_emotion_ids"] = list(chain(*emo))
    instance['token_action_ids'] = list(chain(*acts))
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]  # all -1 except for reply, reply is just the ids
    return instance, sequence


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(topic, history, tokenizer, emotions, actions ,model, args, SPECIAL_TOKENS, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(topic, history, emotions, actions, current_output, tokenizer, SPECIAL_TOKENS, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        token_emotion_ids = torch.tensor(instance['token_emotion_ids'], device = args.device).unsqueeze(0)
        token_action_ids = torch.tensor(instance['token_action_ids'], device = args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids, token_emotion_ids = token_emotion_ids, token_action_ids = token_action_ids)
        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def run():
    config_file = "configs/interact.json"
    config = InteractConfig.from_json_file(config_file)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(config))

    if config.model_checkpoint == "":
        config.model_checkpoint = download_pretrained_model()

    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    logger.info("Get pretrained model and tokenizer")

    logger.info("All Features model interact!!!")
    Special_Tokens = {'bos_token':"<bos>", 'eos_token':"<eos>", 'additional_special_tokens':["<speaker1>","<speaker2>","<no_emotion>", "<happiness>", "<surprise>", "<sadness>", "<disgust>", "<anger>", "<fear>",
                  "<directive>", "<inform>", "<commissive>", "<question>", '<attitude_and_emotion>', '<work>', '<relationship>', '<finance>', '<culture_and_educastion>', '<politics>', '<school_life>', '<tourism>', '<health>', '<ordinary_life>'], 'pad_token':"<pad>"}
    SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>","<speaker2>", 
                  "<no_emotion>", "<happiness>", "<surprise>", "<sadness>", "<disgust>", "<anger>", "<fear>", 
                  '<attitude_and_emotion>', '<work>', '<relationship>', '<finance>', '<culture_and_educastion>', '<politics>', '<school_life>', '<tourism>', '<health>', '<ordinary_life>',
                  "<directive>", "<inform>", "<commissive>", "<question>","<pad>"]

    model_path = 'multi_logger/all_model_128_4_30.bin'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
    tokenizer.add_special_tokens(Special_Tokens)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(model_path))

    model.to(config.device)
    model.eval()

    translator = googletrans.Translator()

    topic = input("topic:>>>")
    topic = tokenizer.encode(topic)
    history = [tokenizer.encode('Hi, what can I help you?')]
    actions = [tokenizer.encode('<inform>')]
    emotions = [tokenizer.encode('<happiness>')]
    # print(translator.translate('Hi, what can I help you?', dest='zh-tw').text)
    while True:
        raw_text = input(">>> ")
        raw_act = input("act:>>> ")
        raw_emo = input("emo:>>> ")
        while not raw_text or not raw_emo or not raw_act:
            print('Prompt should not be empty!')
            raw_text = input("text:>>> ")
            raw_act = input("act:>>> ")
            raw_emo = input("emo:>>> ")

        # raw_text = translator.translate(raw_text, dest = 'en').text
        # print(raw_text)
        if 'bye' in raw_text:
            print('下次見!!!')
            break 

        history.append(tokenizer.encode(raw_text))
        actions.append(tokenizer.encode(raw_act))
        emotions.append(tokenizer.encode(raw_emo))

        with torch.no_grad():
            out_ids = sample_sequence(topic, history, tokenizer, emotions, actions, model, config, SPECIAL_TOKENS)
        history.append(out_ids)
        emotions.append(tokenizer.encode('<happiness>'))
        actions.append(tokenizer.encode('<directive>'))
        history = history[-(2 * config.max_history + 1):]
        emotions = emotions[-(2 * config.max_history + 1):]
        actions = actions[-(2 * config.max_history + 1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)
        # print(translator.translate(out_text, dest = 'zh-tw').text)


if __name__ == "__main__":
    run()