#!/usr/bin/env python
import sys
from constituent import Constituent

class AbstractHeadFindingRules(object):
    def __init__(self):
        self.non_terminal_info = {}
        for line in self.rule_text.strip().split("\n"):
            label, rule = line.strip().split(":", 1)
            self.non_terminal_info[label] = [_.split(",") for _ in rule.split("|")]

    def determine_head(self, t, parent=None):
        if t is None or t.terminal():
            raise Exception("Can't return head of null or left tree.")
        the_head = self.find_marked_head(t)
        if the_head is not None:
            return the_head

        if len(t) == 1:
            return t[0]

        return self.determine_nontrival_head(t, parent)

    def determine_nontrival_head(self, t, parent):
        mother_cat = t.label().split("-")[0]
        how = self.non_terminal_info.get(mother_cat, None)
        if how is None:
            # Should apply default rule.
            raise Exception("No rules match for %s" % mother_cat)

        for idx in xrange(len(how)):
            last_resort = (idx == len(how) - 1)
            the_head = self.travel_locate(t, how[idx], last_resort)
            if the_head is not None:
                break

        return the_head

    def find_marked_head(self, t):
        return None

    def travel_locate(self, t, how, last_resort):
        hid = None
        if how[0] == "left":
            hid = self.find_left_head(t, how)
        elif how[0] == "leftdis":
            hid = self.find_leftdis_head(t, how)
        elif how[0] == "leftexcept":
            hid = self.find_leftexcept_head(t, how)
        elif how[0] == "right":
            hid = self.find_right_head(t, how)
        elif how[0] == "rightdis":
            hid = self.find_rightdis_head(t, how)
        elif how[0] == "rightexcept":
            hid = self.find_rightexcept_head(t, how)
        else:
            raise("ERROR: invalid direction type")

        if hid < 0:
            if last_resort:
                if how[0].startswith("left"):
                    hid = 0
                else:
                    hid = len(t) - 1
                return t[hid]
            else:
                return None

        return t[hid]

    def find_left_head(self, t, how):
        for cat in how[1:]:
            for hid, kid in enumerate(t):
                label = kid.label().split("-")[0]
                if label == cat:
                    return hid
        return -1

    def find_leftdis_head(self, t, how):
        for hid, kid in enumerate(t):
            for cat in how[1:]:
                label = kid.label().split("-")[0]
                if label == cat:
                    return hid
        return -1

    def find_leftexcept_head(self, t, how):
        for hid, kid in enumerate(t):
            found = True
            for cat in how[1:]:
                label = kid.label().split("-")[0]
                if label == cat:
                    found = False
            if found:
                return hid
        return -1

    def find_right_head(self, t, how):
        for cat in how[1:]:
            for hid in xrange(len(t)- 1, -1, -1):
                kid = t[hid]
                label = kid.label().split("-")[0]
                if label == cat:
                    return hid
        return -1

    def find_rightdis_head(self, t, how):
        for hid in xrange(len(t)- 1, -1, -1):
            kid = t[hid]
            for cat in how[1:]:
                label = kid.label().split("-")[0]
                if label == cat:
                    return hid
        return -1

    def find_rightexcept_head(self, t, how):
        for hid in xrange(len(t)- 1, -1, -1):
            kid = t[hid]
            found = True
            for cat in how[1:]:
                label = kid.label().split("-")[0]
                if label == cat:
                    found = False
            if found:
                return hid
        return -1


class CollinsHeadFindingRules(AbstractHeadFindingRules):
    rule_text = '''
ADJP:left,$|rightdis,NNS,NN,JJ,QP,VBN,VBG|left,ADJP|rightdis,JJP,JJR,JJS,DT,RB,RBR,CD,IN,VBD|left,ADVP,NP
JJP:left,NNS,NN,$,QP,JJ,VBN,VBG,ADJP,JJP|JJR,NP,JJS,DT,FW,RBR,RBS,SBAR,RB
ADVP:left,ADVP,IN|rightdis,RB,RBR,RBS,JJ,JJR,JJS|rightdis,RP,DT,NN,CD,NP,VBN,NNP,CC,FW,NNS,ADJP,NML
CONJP:right,CC,RB,IN
FRAG:right
INTJ:left
LST:right,LS,:
NAC:left,NN,NNS,NML,NNP,NNPS,NP,NAC,EX,$,CD,QP,PRP,VBG,JJ,JJS,JJR,ADJP,JJP,FW
PP:right,IN,TO,VBG,VBN,RP,FW,JJ,SYM|left,PP
PRN:left,VP,NP,PP,SQ,S,SINV,SBAR,ADJP,JJP,ADVP,INTJ,WHNP,NAC,VBP,JJ,NN,NNP
PRT:right,RP
QP:left,$,IN,NNS,NN,JJ,CD,PDT,DT,RB,NCD,QP,JJR,JJS
RRC:left,RRC|right,VP,ADJP,JJP,NP,PP,ADVP
S:left,TO,VP,S,FRAG,SBAR,ADJP,JJP,UCP,NP
SBAR:left,WHNP,WHPP,WHADVP,WHADJP,IN,DT,S,SQ,SINV,SBAR,FRAG
SBARQ:left,SQ,S,SINV,SBARQ,FRAG,SBAR
SINV:left,VBZ,VBD,VBP,VB,MD,VBN,VP,S,SINV,ADJP,JJP,NP
SQ:left,VBZ,VBD,VBP,VB,MD,AUX,AUXG,VP,SQ
UCP:right
VP:left,TO,VBD,VBN,MD,VBZ,VB,VBG,VBP,VP,AUX,AUXG,ADJP,JJP,NN,NNS,JJ,NP,NNP
WHADJP:left,WRB,WHADVP,RB,JJ,ADJP,JJP,JJR
WHADVP:right,WRB,WHADVP
WHNP:left,WDT,WP,WP$,WHADJP,WHPP,WHNP
WHPP:right,IN,TO,FW
X:right,S,VP,ADJP,JJP,NP,SBAR,PP,X
NP:rightdis,NN,NNP,NNPS,NNS,NML,NX,POS,JJR|left,NP,PRP|rightdis,$,ADJP,JJP,PRN,FW|right,CD|rightdis,JJ,JJS,RB,QP,DT,WDT,RBR,ADVP
NX:rightdis,NN,NNP,NNPS,NNS,NML,NX,POS,JJR|left,NP,PRP|rightdis,$,ADJP,JJP,PRN,FW|right,CD|rightdis,JJ,JJS,RB,QP,DT,WDT,RBR,ADVP
NML:rightdis,NN,NNP,NNPS,NNS,NML,NX,POS,JJR|left,NP,PRP|rightdis,$,ADJP,JJP,PRN,FW|right,CD|rightdis,JJ,JJS,RB,QP,DT,WDT,RBR,ADVP
POSSP:right,POS
ROOT:left,S,SQ,SINV,SBAR,FRAG
TOP:left,S,SQ,SINV,SBAR,FRAG
TYPO:left,NN,NP,NML,NNP,NNPS,TO,VBD,VBN,MD,VBZ,VB,VBG,VBP,VP,ADJP,JJP,FRAG
ADV:right,RB,RBR,RBS,FW,ADVP,TO,CD,JJR,JJ,IN,NP,NML,JJS,NN
EDITED:left
VB:left,TO,VBD,VBN,MD,VBZ,VB,VBG,VBP,VP,AUX,AUXG,ADJP,JJP,NN,NNS,JJ,NP,NNP
META:left
XS:right,IN
'''
    def __init__(self):
        super(CollinsHeadFindingRules, self).__init__()


class AbstractHeadFinder(object):
    def __init__(self):
        pass

    def run(self, tree):
        '''
        Parameter
        ---------
        tree: nltk.tree.ParentedTree
            The input tree.
        '''
        self.percolate_heads(tree, self.hf)


    def percolate_heads(self, node, hf):
        '''
        Find the head word of the given node.

        Parameter
        ---------
        node: nltk.tree.ParentedTree
            The input phrase head.
        tokens: list
            The output tokens.
        '''
        #print node.pprint(margin=sys.maxint)
        if node.terminal():
            node.head_word = node[0]
        else:
            for kid in node:
                self.percolate_heads(kid, hf)
            head = hf.determine_head(node)
            if head is not None:
                node.head_word = head.head_word
            else:
                print >> sys.stderr, "Head is None:", id(self)


class CollinsHeadFinder(AbstractHeadFinder):
    def __init__(self):
        self.hf = CollinsHeadFindingRules()


if __name__=="__main__":
    #from nltk.tree import ParentedTree
    t = Constituent.fromstring("(S1 (S (NP (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti))) (. .)))")
    head_finder = CollinsHeadFinder()
    head_finder.run(t)

    print "head word is:", t[0].head_word
