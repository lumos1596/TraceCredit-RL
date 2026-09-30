# Manual audit: 16 failed matched teacher-rescue continuations

Population: the 16 non-EM rows among the 21 questions where both the current rescue baseline and the adaptive-query plus evidence-handoff arm are valid. This is a manual path judgment, not a literal-alias score.

| UID | Teacher first prefix judgment | Student continuation | Primary cause |
| --- | --- | --- | --- |
| `2wikimultihopqa:11347` | Correctly finds The Sheltering Desert, but no Chhoti Bahu country evidence. | Searches Chhoti Bahu twice and runs out of suffix. | Incomplete teacher evidence chain. |
| `2wikimultihopqa:12951` | Intended Otakar Vávra death query is right, but top-3 is Jan Svíták/Ester Krumbachová/Czech cinema. | Repeats director lookup, no answer. | Retrieval miss. |
| `2wikimultihopqa:4269` | Charles Chambers birth page is right, Patryk birth fact is absent. | Makes the right Patryk query, gets wrong Patryk Jaki pages, then repeats. | Retrieval miss after a partial prefix. |
| `2wikimultihopqa:655` | Frasquita director query is sensible, but top-3 are unrelated people. | Repeats the same lookup, no answer. | Retrieval miss. |
| `hotpotqa:18953` | Random House Tower is retrieved, but One57 height is absent. | Eventually answers One57. | Semantically correct student answer; strict target stores a long One57 description. |
| `hotpotqa:56526` | Carlos page correctly reaches Édgar Ramírez, but the Gold (2016)/Stephen Gaghan bridge is not retrieved. | Identifies Ramírez, then repeats a broad combined lookup. | Teacher stopped one relation too early. |
| `hotpotqa:63638` | A Tiger Walks producer is supported; Lion King requires one more fact. | Retrieves it and answers Walt Disney Productions. | Semantically correct; strict target is Walt Disney. |
| `hotpotqa:86450` | Ringo Lam birth year is supported; Roberto's is missing. | Retrieves both years, correctly chooses Roberto Benigni. | Semantically correct; target uses full legal name Roberto Remigio Benigni. |
| `musique:11735` | Magdalena Środa biography identifies Poland but not a museum count. | Repeats birthplace lookup. | Teacher does not select the country-to-count bridge. |
| `musique:12868` | Teacher instructs U.S. House representation. | Correctly answers 4 U.S. House seats. | Teacher resolved the wrong sense; target asks 125 Kansas state representatives. |
| `musique:13997` | UNSCR 731 query retrieves unrelated Kazakhstan/Kosovo pages. | Hallucinates a Soviet-space explanation. | Teacher factual/path failure and retrieval miss. |
| `musique:18284` | Album performer → Michael Angelo Batio → Chicago is a correct beginning. | Correctly gets Lake Michigan, then needs Wrigley Field distance and exhausts three searches. | Correct partial teaching, but a three-hop suffix exceeds the available action budget. |
| `musique:1969` | Parliament predecessor query retrieves unrelated municipal/foreign legislative pages. | Guesses Lok Sabha and keeps searching. | Retrieval miss and weak teacher relation decomposition. |
| `musique:2306` | Kanye/Waves pages do not establish the requested musical mechanism. | Answers I Wonder, a song title. | Teacher did not identify the requested relation: melody/chord progression. |
| `musique:6075` | Elvis-file page gives only half the FBI/Ashcroft relation. | Finds enough context and answers Yes. | Semantically correct yes/no answer; stored targets are Elvis entity phrases. |
| `musique:734` | Teacher wrongly identifies the quote as Wilhelm Wundt; it is William James. | Repeats Wundt death-date search and never answers. | Teacher factual-entity error. |

## Attribution

- **4/16 are evaluation-target mismatches:** One57, Walt Disney Productions, Roberto Benigni, and Yes are sensible answers to the user-facing questions but do not exactly match the dataset's stored aliases.
- **11/16 have an inadequate teacher prefix:** six are sensible but incomplete relations or retrieval misses (`11347`, `12951`, `4269`, `655`, `56526`, `11735`); five select a wrong relation/entity or lack relevant evidence (`12868`, `13997`, `1969`, `2306`, `734`).
- **1/16 is chiefly a student suffix budget failure:** `18284` follows the taught chain correctly through two new facts but needs a third fact and distance comparison before answering.

Thus the strict 5/21 number does not mean that 16 students failed to use correct teaching. In this matched set, the dominant controllable problem is teacher path/evidence quality, not the student's failure to imitate a complete correct prefix. The clearest student answer-granularity failure is the unpaired Sar-i Sang case: it explicitly identifies Sar-i Sang in reasoning but answers the broader Badakhshan.

## Implication

Improve teacher rescue in this order:

1. Gate an intervention on whether its top-3 documents cover every unresolved relation, rather than merely whether the query is valid.
2. Ask the teacher to emit an explicit relation chain with a count of remaining hops; intervene later or skip if the remaining chain cannot fit the student action budget.
3. Ask a separate verifier to reject entity/type mistakes such as Wilhelm Wundt versus William James and U.S. House versus Kansas state representatives.
4. Normalize entity aliases and yes/no targets before using rescue EM as the training-selection signal.
