#!/usr/bin/env python3

import sys
import os
import logging

import diff_match_patch as dmp_module


logging.basicConfig()
logger = logging.getLogger(os.path.basename(__file__))
info, warning, error = logger.info, logger.warning, logger.error


dmp = dmp_module.diff_match_patch()

DEL, EQ, INS, OP, TXT = -1, 0, 1, 0, 1   # diff-match-patch constants

DEFAULT_ENCODING = 'utf-8'

ANN_SUFFIX, TXT_SUFFIX = '.ann', '.txt'


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-D', '--database', default=False, action='store_true',
                    help='read and write SQLite DBs instead of files')
    ap.add_argument('-e', '--encoding', default=DEFAULT_ENCODING,
                    help='text encoding (default {})'.format(DEFAULT_ENCODING))
    ap.add_argument('-o', '--output', default=None,
                    help='aligned annotation (default STDOUT)')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='verbose output')
    ap.add_argument('ann', metavar='ANN', help='annotation')
    ap.add_argument('oldtext', metavar='TEXT', help='annotated text')
    ap.add_argument('newtext', metavar='TEXT', help='text to align to')
    return ap


########## annotation ##########


class FormatError(Exception):
    pass


class SpanDeleted(Exception):
    pass


class Annotation(object):
    def __init__(self, id_, type_):
        self.id_ = id_
        self.type_ = type_

    def remap(self, _):
        # assume not text-bound: no-op
        return None

    def fragment(self, _):
        # assume not text-bound: no-op
        return None

    def retext(self, _):
        # assume not text-bound: no-op
        return None


def escape_tb_text(s):
    return s.replace('\n', '\\n')


def is_newline(c):
    # from http://stackoverflow.com/a/18325046
    return c in (
        '\u000A',    # LINE FEED
        '\u000B',    # VERTICAL TABULATION
        '\u000C',    # FORM FEED
        '\u000D',    # CARRIAGE RETURN
        '\u001C',    # FILE SEPARATOR
        '\u001D',    # GROUP SEPARATOR
        '\u001E',    # RECORD SEPARATOR
        '\u0085',    # NEXT LINE
        '\u2028',    # LINE SEPARATOR
        '\u2029'     # PARAGRAPH SEPARATOR
    )


class Textbound(Annotation):
    def __init__(self, id_, type_, offsets, text):
        Annotation.__init__(self, id_, type_)
        self.text = text

        self.offsets = []
        if ';' in offsets:
            # not tested w/discont, so better not to try
            raise NotImplementedError(
                'Discontinuous annotations not supported')
        assert len(offsets) == 2, "Data format error"
        self.offsets.append((int(offsets[0]), int(offsets[1])))

    def remap(self, mapper):
        remapped = []
        for start, end in self.offsets:
            try:
                remapped.append(mapper.remap(start, end))
            except SpanDeleted as e:
                warning('span deleted: {}'.format(e))
        if not remapped:
            raise(SpanDeleted(str(self.offsets)))    # all spans deleted
        self.offsets = remapped

    def fragment(self, text):
        # Remapping may create spans that extend over newlines, which
        # brat doesn't handle well. Break any such span into multiple
        # fragments that skip newlines.
        fragmented = []
        for start, end in self.offsets:
            while start < end:
                while start < end and is_newline(text[start]):
                    start += 1  # skip initial newlines
                fend = start
                while fend < end and not is_newline(text[fend]):
                    fend += 1  # find max sequence of non-newlines
                if fend > start:
                    fragmented.append((start, fend))
                start = fend

        # Switch to fragmented. Edge case: if offsets now only span
        # newlines, replace them with a single zero-length span at
        # the start of the first original span.
        if fragmented:
            self.offsets = fragmented
        else:
            warning('replacing fragmented annotation with zero-width span')
            self.offsets = [(self.offsets[0][0], self.offsets[0][0])]

    def retext(self, text):
        self.orig_text = self.text
        self.text = ' '.join(text[o[0]:o[1]] for o in self.offsets)
        if any(is_newline(c) for c in self.text):
            warning('newline in text: {}'.format(self.text))
        if self.text != self.orig_text:
            if (self.text.replace(' ', '').lower() ==
                self.orig_text.replace(' ', '').lower()):
                log_func = info    # don't warn on space/case change
            else:
                log_func = warning
            log_func('retext: change "{}" to "{}"'.format(
                self.orig_text, self.text))

    def __unicode__(self):
        return "%s\t%s %s\t%s" % (self.id_, self.type_,
                                   ';'.join(['%d %d' % (s, e)
                                             for s, e in self.offsets]),
                                   escape_tb_text(self.text))

    def __str__(self):
        return "%s\t%s %s\t%s" % (self.id_, self.type_,
                                  ';'.join(['%d %d' % (s, e)
                                            for s, e in self.offsets]),
                                  escape_tb_text(self.text))


class XMLElement(Textbound):
    def __init__(self, id_, type_, offsets, text, attributes):
        Textbound.__init__(self, id_, type_, offsets, text)
        self.attributes = attributes

    def __str__(self):
        return "%s\t%s %s\t%s\t%s" % (self.id_, self.type_,
                                      ';'.join(['%d %d' % (s, e)
                                                for s, e in self.offsets]),
                                      escape_tb_text(self.text),
                                      self.attributes)


class ArgAnnotation(Annotation):
    def __init__(self, id_, type_, args):
        Annotation.__init__(self, id_, type_)
        self.args = args


class Relation(ArgAnnotation):
    def __init__(self, id_, type_, args):
        ArgAnnotation.__init__(self, id_, type_, args)

    def __str__(self):
        return "%s\t%s %s" % (self.id_, self.type_, ' '.join(self.args))


class Event(ArgAnnotation):
    def __init__(self, id_, type_, trigger, args):
        ArgAnnotation.__init__(self, id_, type_, args)
        self.trigger = trigger

    def __str__(self):
        return "%s\t%s:%s %s" % (self.id_, self.type_, self.trigger,
                                 ' '.join(self.args))


class Attribute(Annotation):
    def __init__(self, id_, type_, target, value):
        Annotation.__init__(self, id_, type_)
        self.target = target
        self.value = value

    def __str__(self):
        return "%s\t%s %s%s" % (self.id_, self.type_, self.target,
                                '' if self.value is None else ' ' + self.value)


class Normalization(Annotation):
    def __init__(self, id_, type_, target, ref, reftext):
        Annotation.__init__(self, id_, type_)
        self.target = target
        self.ref = ref
        self.reftext = reftext

    def __str__(self):
        return "%s\t%s %s %s\t%s" % (self.id_, self.type_, self.target,
                                     self.ref, self.reftext)


class Equiv(Annotation):
    def __init__(self, id_, type_, targets):
        Annotation.__init__(self, id_, type_)
        self.targets = targets

    def __str__(self):
        return "%s\t%s %s" % (self.id_, self.type_, ' '.join(self.targets))


class Note(Annotation):
    def __init__(self, id_, type_, target, text):
        Annotation.__init__(self, id_, type_)
        self.target = target
        self.text = text

    def __str__(self):
        return "%s\t%s %s\t%s" % (self.id_, self.type_, self.target, self.text)


def parse_xml(fields):
    id_, type_offsets, text, attributes = fields
    type_offsets = type_offsets.split(' ')
    type_, offsets = type_offsets[0], type_offsets[1:]
    return XMLElement(id_, type_, offsets, text, attributes)


def parse_textbound(fields):
    id_, type_offsets, text = fields
    type_offsets = type_offsets.split(' ')
    type_, offsets = type_offsets[0], type_offsets[1:]
    return Textbound(id_, type_, offsets, text)


def parse_relation(fields):
    # allow a variant where the two initial TAB-separated fields are
    # followed by an extra tab
    if len(fields) == 3 and not fields[2]:
        fields = fields[:2]
    id_, type_args = fields
    type_args = type_args.split(' ')
    type_, args = type_args[0], type_args[1:]
    return Relation(id_, type_, args)


def parse_event(fields):
    id_, type_trigger_args = fields
    type_trigger_args = type_trigger_args.split(' ')
    type_trigger, args = type_trigger_args[0], type_trigger_args[1:]
    type_, trigger = type_trigger.split(':')
    return Event(id_, type_, trigger, args)


def parse_attribute(fields):
    id_, type_target_value = fields
    type_target_value = type_target_value.split(' ')
    if len(type_target_value) == 3:
        type_, target, value = type_target_value
    else:
        type_, target = type_target_value
        value = None
    return Attribute(id_, type_, target, value)


def parse_normalization(fields):
    id_, type_target_ref, reftext = fields
    type_, target, ref = type_target_ref.split(' ')
    return Normalization(id_, type_, target, ref, reftext)


def parse_note(fields):
    id_, type_target, text = fields
    type_, target = type_target.split(' ')
    return Note(id_, type_, target, text)


def parse_equiv(fields):
    id_, type_targets = fields
    type_targets = type_targets.split(' ')
    type_, targets = type_targets[0], type_targets[1:]
    return Equiv(id_, type_, targets)


parse_standoff_func = {
    'T': parse_textbound,
    'R': parse_relation,
    'E': parse_event,
    'N': parse_normalization,
    'M': parse_attribute,
    'A': parse_attribute,
    'X': parse_xml,
    '#': parse_note,
    '*': parse_equiv,
}


def parse_standoff_line(l, ln, fn):
    try:
        return parse_standoff_func[l[0]](l.split('\t'))
    except Exception:
        raise FormatError("error on line {} in {}: {}".format(ln, fn, l))


def parse_ann_file(fn, options):
    annotations = []
    with open(fn, 'r', encoding=options.encoding) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            if not l or l.isspace():
                continue
            annotations.append(parse_standoff_line(l, ln, fn))
    return annotations


def parse_ann_data(data, name, options):
    annotations = []
    for ln, l in enumerate(data.split('\n'), start=1):
        l = l.rstrip('\n')
        if not l or l.isspace():
            continue
        annotations.append(parse_standoff_line(l, ln, name))
    return annotations


def is_word_start(offset, text):
    # Accept alnum/non-alnum as well as any non-space/space (or doc
    # start) as word boundary
    if text[offset].isalnum():
        return offset == 0 or not text[offset-1].isalnum()
    elif not text[offset].isspace():
        return offset == 0 or text[offset-1].isspace()
    else:
        return False


def is_word_end(offset, text):
    # Analogous to is_word_start
    if offset == 0:
        return False
    elif text[offset-1].isalnum():
        return offset == len(text) or not text[offset].isalnum()
    elif not text[offset-1].isspace():
        return offset == len(text) or text[offset].isspace()
    else:
        return False


class Remapper(object):
    def __init__(self, old_text, new_text, offset_map):
        self.old_text = old_text
        self.new_text = new_text
        self.offset_map = offset_map

    def _remap(self, start, end):
        if start == end:
            return self.offset_map[start], self.offset_map[end]    # empty span
        elif self.offset_map[start] == self.offset_map[end]:
            # non-empty span maps to empty
            raise SpanDeleted('{}:{} -> {}:{}'.format(
                start, end, self.offset_map[start], self.offset_map[end]))
        else:
            return self.offset_map[start], self.offset_map[end - 1] + 1

    def remap(self, start, end, max_realign_distance=10):
        new_start, new_end = self._remap(start, end)
        if not max_realign_distance:
            return new_start, new_end

        # Don't attempt to modify zero-length annotations
        if start == end or new_start == new_end:
            return new_start, new_end

        # if span started/ended at word boundary before, try to
        # re-identify the boundary by extending the span.
        # TODO: consider checking if chars before/after span match
        old_text, new_text = self.old_text, self.new_text
        old_span_text = old_text[start:end]
        re_start = None
        if is_word_start(start, old_text):
            for i in range(max_realign_distance):
                if new_start-i >= 0 and is_word_start(new_start-i, new_text):
                    re_start = new_start - i
                    break
        # Also, try to strip initial space if there wasn't any in the
        # original
        if not old_text[start].isspace() and new_text[new_start].isspace():
            for i in range(1, max_realign_distance):
                if (new_start+i < new_end and
                    not new_text[new_start+i].isspace()):
                    re_start = new_start + i
                    break
        if re_start is not None and re_start != new_start:
            warning('realign: "{}" to "{}" (original "{}")'.format(
                new_text[new_start:new_end], new_text[re_start:new_end],
                old_span_text))
            new_start = re_start
        re_end = None
        if is_word_end(end, self.old_text):
            for i in range(max_realign_distance):
                if new_end+i < len(new_text) and is_word_end(new_end+i,
                                                             new_text):
                    re_end = new_end + i
                    break
        if not old_text[end-1].isspace() and new_text[new_end-1].isspace():
            for i in range(1, max_realign_distance):
                if (new_end-1-i > new_start and
                    not new_text[new_end-1-i].isspace()):
                    re_end = new_end - i
                    break
        if re_end is not None and re_end != new_end:
            warning('realign: "{}" to "{}" (original "{}")'.format(
                new_text[new_start:new_end], new_text[new_start:re_end],
                old_span_text))
            new_end = re_end
        return new_start, new_end


########## diff ##########


def alignment_strings(diff):
    a1, a2 = [], []
    for op, s in diff:
        if op == EQ:
            a1.append(s)
            a2.append(s)
        elif op == DEL:
            a1.append(s)
            a2.append('-'*len(s))
        elif op == INS:
            a1.append('-'*len(s))
            a2.append(s)
    return ''.join(a1), ''.join(a2)


def diff_to_offset_map(diff):
    o, ins_count, del_count, offset_map = 0, 0, 0, []
    for op, s in diff:
        if op == EQ:
            # resolve possible previous deletes and inserts
            sub_count = min(ins_count, del_count)
            # in substitutions, arbitrarily match initial characters
            for i in range(sub_count):
                offset_map.append(o)
                o += 1
            ins_count -= sub_count
            del_count -= sub_count
            if del_count:
                # deletion: next characters all map to the same offset
                for i in range(del_count):
                    offset_map.append(o)
                del_count = 0
            if ins_count:
                # insertion: update offset
                o += ins_count
                ins_count = 0
            for i in range(len(s)):
                offset_map.append(o+i)
            o += len(s)
        elif op == DEL:
            del_count += len(s)
        elif op == INS:
            ins_count += len(s)
    # resolve leftovers
    sub_count = min(ins_count, del_count)
    for i in range(sub_count):
        # trailing substitution: arbitrarily match initial characters
        offset_map.append(o)
        o += 1
    ins_count -= sub_count
    del_count -= sub_count
    if del_count > 0:
        # trailing delete: remaining text maps to last offset
        for i in range(del_count):
            offset_map.append(o)
    return offset_map


def words_to_chars(words, word_list, word_hash):
    # Adapted from diff_match_patch.py diff_linesToCharsMunge()
    # "\x00" is a valid character, but various debuggers don't like it.
    # So we'll insert a junk entry to avoid generating a null character.
    if not(word_list):
        word_list.append('')
    chars = []
    for word in words:
        if word not in word_hash:
            if len(word_list) >= sys.maxunicode:
                raise ValueError('too many unique words')
            word_hash[word] = len(word_list)
            word_list.append(word)
        chars.append(chr(word_hash[word]))
    return ''.join(chars)


def word_diff(words1, words2):
    # Adapted from diff_match_patch.py diff_linesToChars/charsToLines()
    word_list, word_hash = [], {}

    # replace each word with a unicode character
    chars1 = words_to_chars(words1, word_list, word_hash)
    chars2 = words_to_chars(words2, word_list, word_hash)

    # take a word-level diff
    diff = dmp.diff_main(chars1, chars2, checklines=False)

    # replace chars back with words
    for i in range(len(diff)):
        text = []
        for char in diff[i][1]:
            text.append(word_list[ord(char)])
        diff[i] = (diff[i][OP], ''.join(text))

    return diff


def re_diff(diff):
    # Adapted from diff_match_patch.py diff_lineMode()
    # Re-diff word (or sentence) level substitutions on character level
    diff.append((EQ, ''))    # add sentinel
    del_count, ins_count = 0, 0
    del_text, ins_text = '', ''
    i = 0
    while i < len(diff):
        if diff[i][0] == INS:
            ins_count += 1
            ins_text += diff[i][TXT]
        elif diff[i][0] == DEL:
            del_count += 1
            del_text += diff[i][TXT]
        elif diff[i][0] == EQ:
            # check for substitutions (delete and insert)
            if del_count >= 1 and ins_count >= 1:
                # delete substitution, add char-level diff
                sub_diff = dmp.diff_main(del_text, ins_text, False)
                diff[i-(del_count+ins_count):i] = sub_diff
                i += len(sub_diff) - (del_count + ins_count)
            del_count, ins_count = 0, 0
            del_text, ins_text = '', ''
        i += 1
    diff.pop()    # drop sentinel
    return diff


def has_initial_space(s, pos=0):
    return pos < len(s) and s[pos].isspace()


def initial_space_count(s, start=0):
    """Return number of initial spaces in s."""
    i, end = 0, len(s)
    while start+i < end and s[start+i].isspace():
        i += 1
    return i


def find_end_ignorespace(s, sub, start=0):
    """Find end of substring sub in s, ignoring extra space in s."""
    i, j = 0, 0
    while j < len(sub):
        if s[start+i] == sub[j]:
            i += 1
            j += 1
        elif s[start+i].isspace():
            i += 1
        else:
            raise ValueError('cannot find substring "{}" in "{}"'.\
                             format(sub, s[start:]))
    # TODO skip trailing space too?
    return start+i


def diff_ignorespace(text1, text2):
    # Note: tokenization that splits punctuation (etc.) might produce
    # better diffs than whitespace splitting in some cases.
    words1 = text1.split()
    words2 = text2.split()

    # start with word-level diff
    diff = word_diff(words1, words2)

    # TODO: consider calling dmp.diff_cleanupSemantic(diff) here

    # extend to character level
    diff = re_diff(diff)

    # helper, avoids code duplication
    def _resolve_initial_space(text1, text2, o1, o2, diff, i=None):
        if i is None:
            i = len(diff)    # append
        s1 = initial_space_count(text1, o1)
        s2 = initial_space_count(text2, o2)
        if s1 and s2:
            # space in both, add EQ spanning as much as possible
            sb = min(s1, s2)
            diff.insert(i, (EQ, text1[o1:o1+sb]))
            o1 += sb
            o2 += sb
        elif s1:
            # space only in text1, add DEL removing it
            diff.insert(i, (DEL, text1[o1:o1+s1]))
            o1 += s1
        elif s2:
            # space only in text2, add INS adding it
            diff.insert(i, (INS, text2[o2:o2+s2]))
            o2 += s2
        return o1, o2, diff

    # Put space back in. Note that this doesn't make a diff for space:
    # space substitutions such as ' ' for '\t' are ignored.
    o1, o2, i = 0, 0, 0
    while i < len(diff):
        if has_initial_space(text1, o1) or has_initial_space(text2, o2):
            o1, o2, diff = _resolve_initial_space(text1, text2, o1, o2, diff, i)
        else:
            # no initial space, process diff and put space back in
            if diff[i][OP] == EQ:
                e1 = find_end_ignorespace(text1, diff[i][TXT], o1)
                e2 = find_end_ignorespace(text2, diff[i][TXT], o2)
                sub_diff = dmp.diff_main(text1[o1:e1], text2[o2:e2])
                diff[i:i+1]= sub_diff
                i += len(sub_diff)-1
                o1 = e1
                o2 = e2
            elif diff[i][OP] == DEL:
                e1 = find_end_ignorespace(text1, diff[i][TXT], o1)
                diff[i] = (diff[i][OP], text1[o1:e1])
                o1 = e1
            else:
                e2 = find_end_ignorespace(text2, diff[i][TXT], o2)
                diff[i] = (diff[i][OP], text2[o2:e2])
                o2 = e2
        i += 1

    # handle space at end
    while has_initial_space(text1, o1) or has_initial_space(text2, o2):
        o1, o2, diff = _resolve_initial_space(text1, text2, o1, o2, diff)

    return diff


def align(annotations, old_text, new_text, ann_name, old_name, new_name,
          options):
    diff = diff_ignorespace(old_text, new_text)

    # verbose diagnostic output
    distance = dmp.diff_levenshtein(diff)
    info('{} <> {} distance {} (lengts {}, {})'.format(
        old_name, new_name, distance, len(old_text), len(new_text)))
    for a in alignment_strings(diff):
        info('alignment:\n{}'.format(a))

    offset_map = diff_to_offset_map(diff)
    assert len(offset_map) == len(old_text), 'internal error: {} {}'.format(len(offset_map), len(old_text))

    mapper = Remapper(old_text, new_text, offset_map)
    for a in annotations:
        try:
            a.remap(mapper)
        except SpanDeleted:
            warning('annotation deleted from {}: {}'.format(ann_name, a))
            continue
        a.fragment(new_text)
        a.retext(new_text)

    return annotations


def write_annotations(annotations, out):
    for a in annotations:
        print(str(a), file=out)


def align_files(ann_file, old_file, new_file, options):
    annotations = parse_ann_file(ann_file, options)

    with open(old_file, encoding=options.encoding) as f:
        old_text = f.read()
    with open(new_file, encoding=options.encoding) as f:
        new_text = f.read()

    annotations = align(annotations, old_text, new_text,
                        ann_file, old_file, new_file, options)

    if options.output is None:
        write_annotations(annotations, sys.stdout)
    else:
        with open(options.output, 'wt', encoding=options.encoding) as out:
            write_annotations(annotations, out)


def align_dbs(ann_db_path, old_db_path, new_db_path, options):
    try:
        from sqlitedict import SqliteDict
    except ImportError:
        error('failed `import sqlitedict`, try `pip3 install sqlitedict`')
        raise

    if ann_db_path != old_db_path:
        raise NotImplementedError()    # TODO

    if not os.path.exists(ann_db_path):
        raise IOError('no such file: {}'.format(ann_db_path))

    if not os.path.exists(new_db_path):
        raise IOError('no such file: {}'.format(new_db_path))

    insert_count = 0
    with SqliteDict(ann_db_path, flag='r') as from_db:
        with SqliteDict(new_db_path, flag='r') as to_db:
            with SqliteDict(options.output, autocommit=False) as out_db:
                for key, value in from_db.items():
                    root, suffix = os.path.splitext(key)
                    if suffix != ANN_SUFFIX:
                        continue
                    ann_key, ann_data = key, value
                    text_key = root+TXT_SUFFIX

                    old_text = from_db.get(text_key)
                    if old_text is None:
                        warning('{} not found in {}, skipping {}'.format(
                            text_key, ann_db_path, key))
                        continue

                    new_text = to_db.get(text_key)
                    if new_text is None:
                        warning('{} not found in {}, skipping {}'.format(
                            text_key, new_db_path, key))
                        continue

                    annotations = parse_ann_data(ann_data, ann_key, options)

                    annotations = align(annotations, old_text, new_text,
                                        ann_key, text_key, text_key, options)

                    ann_str = '\n'.join(str(a) for a in annotations)
                    out_db[ann_key] = ann_str

                    insert_count += 1
                    if insert_count % 1000 == 0:
                        print('Inserted {}, committing...'.format(
                            insert_count, end='', file=sys.stderr, flush=True))
                        out_db.commit()
                        print('done.', file=sys.stderr)
                out_db.commit()


def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.database and not args.output:
        error('argument --output required with --database')
        return 1
    if args.verbose:
        logger.setLevel(logging.INFO)

    try:
        if not args.database:
            align_files(args.ann, args.oldtext, args.newtext, args)
        else:
            align_dbs(args.ann, args.oldtext, args.newtext, args)
    except Exception as e:
        error('failed {} {} {} {}'.format(
            __file__, args.ann, args.oldtext, args.newtext))
        raise

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
