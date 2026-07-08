# Results

> **Plain-English summary:** after fine-tuning, the model's responses got
> noticeably more "dialect-flavored" — shorter, and using Najdi words and
> phrasing instead of formal textbook Arabic — on the same three
> government-service questions. The training numbers behave exactly like
> you'd expect for a small fine-tune: loss goes down steadily on the
> training data, while the validation loss improves once and then ticks
> back up slightly, a normal early sign that the model has learned about as
> much as this small dataset can teach it.

## Training loss

| Epoch | Training loss |
|---|---|
| 0 | 2.037 |
| 1 | 1.746 |
| 2 | 1.517 |

## Validation loss

| Epoch | Validation loss |
|---|---|
| 0 | 1.865 |
| 1 | 1.810 |
| 2 | 1.821 |

Training loss decreases monotonically across all 3 epochs, as expected.
Validation loss improves from epoch 0 to epoch 1, then rises slightly at
epoch 2 (1.810 → 1.821) even as training loss keeps falling (1.746 →
1.517). This divergence is a classic early-overfitting signal: by epoch 2
the model is fitting training-specific phrasings more than it's improving
general Najdi government-services response quality. The adapter checkpoint
used for evaluation below is the epoch 2 checkpoint (final), since the
qualitative gap between epoch 1 and epoch 2 validation loss is small (0.011)
and epoch 2 showed clearly more consistent dialect usage in spot checks —
see [Limitations](#honest-limitations) for the dataset-size fix that would
let us resolve this more rigorously (e.g. by selecting epoch 1 as the
checkpoint, or adding regularization) instead of eyeballing it.

## Side-by-side outputs

All three questions are from the fixed evaluation set in
`scripts/evaluate.py` / notebook Cell 9. Outputs are representative samples
from running the fine-tuned adapter's epoch-2 checkpoint; exact wording
varies run to run due to sampling (`temperature=0.7`, `top_p=0.9`).

### Question 1

**Najdi prompt:** وش الأوراق اللي أحتاجها عشان أجدد الإقامة؟
*("What documents do I need to renew my residency permit?")*

| Base ALLaM-7B | Fine-tuned (this project) |
|---|---|
| تحتاج إلى تقديم مجموعة من الوثائق لتجديد الإقامة، وتشمل: جواز السفر ساري المفعول، الإقامة الحالية، شهادة عدم وجود مخالفات مرورية، بالإضافة إلى سداد الرسوم المقررة عبر منصة أبشر. يُنصح بمراجعة الموقع الرسمي للتأكد من آخر المتطلبات. | تحتاج جواز ساري وصورة من الإقامة القديمة، وتسدد الرسوم في أبشر. الحين صار كله أونلاين، ما تحتاج تروح لمكتب. لو عندك مخالفات لازم تسويها أول. |

### Question 2

**Najdi prompt:** كيف أقدر أحجز موعد في الأحوال المدنية؟
*("How can I book an appointment at Civil Affairs?")*

| Base ALLaM-7B | Fine-tuned (this project) |
|---|---|
| يمكنك حجز موعد في الأحوال المدنية من خلال منصة أبشر الإلكترونية، حيث تقوم بتسجيل الدخول إلى حسابك، ثم اختيار خدمة "حجز موعد"، وتحديد الفرع والتاريخ والوقت المناسبين، وذلك وفق الخطوات التالية... | ادخل أبشر واختر "حجز موعد"، وحدد الفرع اللي يناسبك والوقت. لو ما لقيت موعد قريب جرب فروع ثانية قريبة منك، أحيانًا يكون فيها مواعيد أسرع. |

### Question 3

**Najdi prompt:** ودي أستفسر عن رسوم تجديد رخصة القيادة، كم تكلفتها؟
*("I'd like to ask about driving license renewal fees, how much do they cost?")*

| Base ALLaM-7B | Fine-tuned (this project) |
|---|---|
| تختلف رسوم تجديد رخصة القيادة حسب المدة المختارة للتجديد، حيث تتراوح الرسوم بين مبالغ محددة لكل من التجديد لمدة سنة أو خمس سنوات أو عشر سنوات. يُرجى مراجعة منصة أبشر للاطلاع على القيم الدقيقة والمحدثة. | تعتمد على المدة اللي تبيها، سنة أو خمس أو عشر سنين، وكل مدة لها رسومها. أبشر يبين لك السعر بالضبط أول ما تدخل تجدد، قبل لا تدفع. |

## Analysis

The fine-tuned model consistently shows three changes relative to base
ALLaM-7B on these prompts:

1. **Dialect function words.** The fine-tuned outputs use Najdi markers
   that the base model's MSA-leaning responses avoid: **الحين** ("now",
   vs. MSA حاليًا/الآن), **وش/إيش**-register phrasing carried over from the
   question into the response's implicit register, **أبيك/أبيه**-style verb
   forms are avoided in favor of direct imperative/conversational
   constructions ("ادخل أبشر واختر...", "تسدد الرسوم..."), and **لو**
   used colloquially for "if" in a casual register rather than the more
   formal إذا preferred by the base model.
2. **Shorter, more conversational structure.** Base responses read like
   excerpted government FAQ text — enumerated, formal, hedged ("يُنصح
   بمراجعة الموقع الرسمي"). Fine-tuned responses read like a person
   answering a friend's question directly, dropping the throat-clearing
   and getting to the actionable point faster.
3. **Practical, experience-based additions.** Notably in Question 2, the
   fine-tuned response adds a practical tip not present in the base
   response ("جرب فروع ثانية قريبة منك") — a register of helpfulness more
   typical of the training data's customer-service-style conversations
   than of formal government copy.

Together these are consistent with what LoRA fine-tuning on a filtered,
dialect-native conversational dataset is expected to do: shift response
*register and phrasing* without teaching new factual content (the
underlying facts — which documents, which platform, which fee tiers — are
the same in both columns).

## Honest limitations

- **Responses are shorter than the base model's.** In all three examples
  above, the fine-tuned model's responses are noticeably terser than the
  base model's. This is very likely because the filtered training subset
  (`government_services` + `customer_service` conversations from
  `HeshamHaroon/saudi-dialect-conversations`) skews toward short,
  conversational responses, and the model is fitting that response-length
  distribution along with the dialect register. For some real deployment
  use cases the shorter answers may under-explain (e.g. omitting the "check
  Absher for exact numbers" caveat that the base model includes).
- **Next step: a larger, more diverse dataset.** The filtered training set
  is a small slice of one source dataset. The validation loss upturn at
  epoch 2 (see [Training loss](#training-loss) /
  [Validation loss](#validation-loss)) is consistent with a dataset too
  small to fully separate "learning Najdi register" from "memorizing this
  dataset's specific response patterns and lengths." Expanding the training
  set — more government-services dialogue, and/or blending in longer-form
  dialect examples to counteract the length-shrinkage effect above — is the
  clearest next step toward a more robust adapter.
