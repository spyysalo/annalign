# annalign

Align brat-flavored standoff annotations to similar text


## Quickstart

```
pip3 install diff-match-patch

python3 annalign.py \
    examples/28439270.old.ann \
    examples/28439270.old.txt \
    examples/28439270.new.txt \
  > examples/28439270.new.ann

diff examples/28439270.old.ann examples/28439270.new.ann
```


## What is this?

A tool for working with text annotations in the
[brat-flavored standoff format](http://brat.nlplab.org/standoff).
Given two largely identical texts, `annalign.py` maps offsets and
text references in standoff data from one to the other.

This type of mapping is needed e.g. to update annotations to changes
in source text and to compare annotations for different versions of
the same text.


## Acknowledgments

Derived in part from brat
[annalign.py](https://github.com/nlplab/brat/blob/master/tools/annalign.py)
and in part from
[https://github.com/google/diff-match-patch](diff-match-patch).
