%% onera.bst par F. CASSAING DOTA
FUNCTION {version} {
"onera.bst v3.7 21/09/2006, bugreports: frederic.cassaing@onera.fr"}
%%
%% KESAKO ?
%% Feuille de style compatible avec la charte ONERA pour l'utilitaire bibtex
%% (creation des references bibliographiques pour latex a partir de bases 
%% de donnees bibliographiques *.bib et de cles d'appel dans fichier latex).
%% Le principe de bibtex est expose sur la page d'aide Latex (http://boa/latex)
%% Pour obtenir la derniere version sur boa: ftp://125.1.4.53/pub/tex/contrib 
%%
%% MINIDOC TECHNIQUE
%% L'interligne entre references bibliographiques (parametre latex) est pris
%% en compte a partir de la version 2.3 de onera.sty (de Laurent Mugnier).
%% Le format "ouvert" de la charte ONERA suppose l'utilisation complementaire
%% du package "openbib" (inclus par onera.cls a partir de la version 3)
%% Toutes les fonctions sont definies en dur, donc inutile d'utiliser frbib.sty
%% Les termes francais (mai, rapport technique, editeur) sont automatiquement
%% convertis en anglais si l'option english est passee a la classe onera
%% On peut aussi les changer a la main (autres langues...). Pour cela, recopier
%% les declarations au debut du .bbl ds le .tex, avant les \biblio..., en
%% remplacant les "\providecommand{\Xname}{}" par "\newcommand{\Xname}{toto}
%% Cette feuille de style a ete obtenue par l'utilitaire makebst, et completee
%% manuellement: francisation/englicisation, format onera, cesure nom propres,
%%               utilitaires (version, comptage des refs), ponctuations

% Ajout (par rapport au format bibtex standard) d'une fonction pour declarer le
% preamble en debut de doc. But: francisation CHANGEABLE depuis doc principal, 
% et creation d'une macro \onerabstbox[toto] (1 arg optionel) pour remplacer
% \mbox{toto}. Le but est de mettre les noms propres en boite ds format.name$
% pour eviter leur cesure sans utiliser de {}, interdit ds format.name$, en 
% les remplacant par [] grace a l'argument optionel

% Bibtex semble mettre une limite a la taille des fonctions => compactage
FUNCTION {bstpreamble} {
  "\providecommand{\onerabstbox}[1][]{\mbox{#1}} "
  "\makeatletter " *
  "\providecommand{\ifONERA@english}{\ifx10} " *  % execution hors onera.cls
  "\ifONERA@english " *
  "\providecommand{\janname}{January}\providecommand{\febname}{February} " * 
  "\providecommand{\marname}{March}\providecommand{\aprname}{April} " * 
  "\providecommand{\mayname}{May}\providecommand{\junname}{June} " * 
  "\providecommand{\julname}{July}\providecommand{\augname}{August} " * 
  "\providecommand{\sepname}{September}\providecommand{\octname}{October} " * 
  "\providecommand{\novname}{November}\providecommand{\decname}{December} " * 
  "\providecommand{\Inname}{In}\providecommand{\inname}{in} " * 
  "\providecommand{\andname}{and}\providecommand{\ofname}{of} " * 
  "\providecommand{\techreportname}{Technical report} " * 
  "\providecommand{\masterthesisname}{Rapport de fin d'\'etudes} " * 
  "\providecommand{\phdthesisname}{PhD thesis} " *
  "\providecommand{\editorname}{editor} " * 
  "\providecommand{\editorsname}{editors} " * 
  "\providecommand{\editionname}{edition} " * %en anglais comme change.case$
  "\providecommand{\editedbyname}{edited by} " * 
  "\else " *
  "\providecommand{\janname}{janvier}\providecommand{\febname}{f\'evrier} " * 
  "\providecommand{\marname}{mars}\providecommand{\aprname}{avril} " * 
  "\providecommand{\mayname}{mai}\providecommand{\junname}{juin} " * 
  "\providecommand{\julname}{juillet}\providecommand{\augname}{ao\^ut} " * 
  "\providecommand{\sepname}{septembre}\providecommand{\octname}{octobre} " * 
  "\providecommand{\novname}{novembre}\providecommand{\decname}{d\'ecembre} " *
  "\providecommand{\Inname}{Dans}\providecommand{\inname}{dans} " * 
  "\providecommand{\andname}{et}\providecommand{\ofname}{de} " * 
  "\providecommand{\techreportname}{Rapport Technique} " * 
  "\providecommand{\masterthesisname}{Rapport de fin d'\'etudes} " * 
  "\providecommand{\phdthesisname}{Th\`ese de doctorat} " *
  "\providecommand{\editorname}{r\'edacteur en chef} " * 
  "\providecommand{\editorsname}{r\'edacteurs en chef} " * 
  "\providecommand{\editionname}{\'edition} " *
  "\providecommand{\editedbyname}{sous la direction de} " * 
  "\fi\makeatother\hbadness=9000" *     % reduit underful box suite anticesure
}


ENTRY
  { address
    author
    booktitle
    chapter
    edition
    editor
    howpublished
    institution
    journal
    key
    lang    % ajout test le 23/9/2002
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    volume
    year
  }
  {}
  { label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block
nb.refs }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
  #0 'nb.refs :=
}

STRINGS { s t } %output.lang test}

FUNCTION {word.in}{ "\Inname{} " }

FUNCTION {word.and}{ " \andname{} " }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { %  add.period$ write$      % add.period$ enleve le 17/9/99
		  "," * write$               % remplace par une virgule
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ t " vide dans " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {fin.entry}
{ duplicate$ empty$
    'pop$
     { add.period$ write$ }
  if$
  newline$
%  "\par ~" write$ newline$	% saut de ligne en fin de ref, enleve par openbib
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {non.stop}
{ duplicate$
   "}" * add.period$
   #-1 #1 substring$ "." =
}

FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}

FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}

FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}

FUNCTION {new.sentence.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.sentence
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize} 
{ duplicate$ empty$
    { pop$ "" }
    { "{\it " swap$ * non.stop
        { "\/}" * }
        { "}" * }
      if$
    }
  if$
}

INTEGERS { nameptr namesleft numnames }

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{f.~}{vv~}\onerabstbox[{ll}]{~jj}" format.name$ 't :=  %v3.0:anticesure
      nameptr #1 >
        {
          namesleft #1 >
            { ", " * t * }
            { t "\onerabstbox[others]" =
                { " " * "et~al." * }
                { word.and * t * }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.names.ed}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{f.~}{vv~}\onerabstbox[{ll}]{~jj}" format.name$ 't := %v3.0:anticesure
      nameptr #1 >
        {
          namesleft #1 >
            { ", " * t * }
            { t "\onerabstbox[others]" =
                { " " * "et~al." * }
                { word.and * t * }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names }
  if$
}

FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.names
      editor num.names$ #1 >
%        { ", \'editeurs" * }
%        { ", \'editeur" * }
        { ", \editorsname{}" * }
        { ", \editorname{}" * }
      if$
    }
  if$
}

FUNCTION {format.in.editors}
{ editor empty$
    { "" }
    { editor format.names.ed
    }
  if$
}

FUNCTION {format.title}
{ title empty$
    { "" }
    { title
      emphasize
      % add.period$    % Ajoute version 3.0 le 10/12/99
    }
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
  nb.refs #1 + 'nb.refs :=
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {conv.month}    
% Transforme le mois (en haut de la pile) de "1" a "janvier", etc
% ce n'est pas le codage officiel, mais si utilised, autant convertir !!
{ duplicate$ "1" =
    { pop$ "@jan" }
	'skip$
  if$		
  duplicate$ "2" =
    { pop$ "@feb" }
	'skip$
  if$		
  duplicate$ "3" =
    { pop$ "@mar" }
	'skip$
  if$		
  duplicate$ "4" =
    { pop$ "@apr" }
	'skip$
  if$		
  duplicate$ "5" =
    { pop$ "@may" }
	'skip$
  if$		
  duplicate$ "6" =
    { pop$ "@jun" }
	'skip$
  if$		
  duplicate$ "7" =
    { pop$ "@jul" }
	'skip$
  if$		
  duplicate$ "8" =
    { pop$ "@aug" }
	'skip$
  if$		
  duplicate$ "9" =
    { pop$ "@sep" }
	'skip$
  if$		
  duplicate$ "10" =
    { pop$ "@oct" }
	'skip$
  if$		
  duplicate$ "11" =
    { pop$ "@nov" }
	'skip$
  if$		
  duplicate$ "12" =
    { pop$ "@dec" }
	'skip$
  if$		
}

FUNCTION {format.date}
{ year empty$
    { month empty$
        { "" }
        { "Mois sans annee dans " cite$ * warning$
          month
        }
      if$
    }
    {
      before.all 'output.state :=
      " (" 
      month empty$
	    'skip$
	    { month conv.month
		  * " " * 
        }
      if$
    year * ")" *
    }
  if$
}

FUNCTION {format.btitle}
{ title  emphasize
}

FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "Redondance " swap$ * " fields in " * cite$ * warning$ }
% msg orig : can't use both 
  if$
}

FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { output.state mid.sentence =
        { "vol." }
        { "Vol." }
      if$
      volume tie.or.space.connect
      series empty$
        'skip$
        { " \ofname{} " * series emphasize  * }
      if$
      "volume and number" number either.or.check
    }
  if$
}

FUNCTION {format.number.series}
{ volume empty$
    { number empty$
        { series field.or.null }
        { output.state mid.sentence =
            { "no." }
            { "No." }
          if$
          number tie.or.space.connect
          series empty$
            { "Pour info : il y a un 'number' mais pas de 'series' dans " 
               cite$ * warning$ }
            { " \inname{} " * series * }
          if$
        }
      if$
    }
    { "" }
  if$
}

FUNCTION {format.edition}
{ edition empty$
    { "" }
    { output.state mid.sentence =
        { edition "l" change.case$ " " * "\editionname{}" * }
        { edition "t" change.case$ " " * "\editionname{}" * }
      if$
    }
  if$
}

INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages multi.page.check
        { "pp.~" pages n.dashify *} %V3.0 tie.or.space.connect remplace par ~*
        { "p.~" pages *}  %V3.0 tie.or.space.connect remplace par ~ et *
      if$
    }
  if$
}

FUNCTION {format.vol.num}
{ volume field.or.null
  number empty$
    'skip$
    { " (" number * ")" * *
      volume empty$
        { "Il y a un number mais pas de volume dans " cite$ * warning$ }
        'skip$
      if$
    }
  if$
}

FUNCTION {format.vol.num.pages}
{ volume field.or.null
  number empty$
    'skip$
    { " (" number * ")" * *
      volume empty$
        { "Il y a un number mais pas de volume dans " cite$ * warning$ }
        'skip$
      if$
    }
  if$
  pages empty$
    'skip$
    { duplicate$ empty$
        { pop$ format.pages }
        { "~: " * pages n.dashify * }
      if$
    }
  if$
}

FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
        { output.state mid.sentence =  % Ajoute le 10/12/99 au lieu 
            { "chap." }                % du seul "chap. " present 
            { "Chap." }                % precedemment
        if$ }                          %
        { type "l" change.case$ }
      if$
      chapter tie.or.space.connect
      pages empty$
        'skip$
        { ", " * format.pages * }
      if$
    }
  if$
}

FUNCTION {format.in.ed.booktitle}
{ booktitle empty$
    { "" }
    { editor empty$
        { word.in booktitle emphasize * }
%        { word.in booktitle emphasize * ", \'edit\'e par " *
        { word.in booktitle emphasize * ", \editedbyname{} " *
          format.in.editors * }
      if$
    }
  if$
}

FUNCTION {empty.misc.check}
{ author empty$ title empty$ howpublished empty$
  month empty$ year empty$ note empty$
  and and and and and
  key empty$ not and
    { "tous les champs necessaires sont vides dans " cite$ * warning$ }
    'skip$
  if$
}

FUNCTION {format.thesis.type}
{ type empty$
    'skip$
    { pop$
      type "t" change.case$
    }
  if$
}

FUNCTION {format.tr.number}
{ type empty$
    { "\techreportname{}" }
    'type
  if$
  institution empty$
    'skip$
    {" " institution * *}
  if$
  number empty$
    {  }              % modif v1.9 du 03/04/97 : removed "t" change.case$
    { " \no" * number tie.or.space.connect}
  if$
}

FUNCTION {format.article.crossref}
{
  key empty$
    { journal empty$
        { "manque key ou journal pour " cite$ * 
          " pour r\'ef crois\'ee" * crossref *
          warning$
          ""
        }
        { word.in journal emphasize * }
      if$
    }
    { word.in key * " " *}
  if$
  "\cite{" * crossref * "}" *
}

FUNCTION {format.crossref.editor}
{ editor #1 "{vv~}\onerabstbox[{ll}]" format.name$ %v3.0:anticesure
  editor num.names$ duplicate$
  #2 >
    { pop$ " " * "et~al." emphasize * }
    { #2 <
        'skip$
        {editor #2 "{f. }{vv }\onerabstbox[{ll}]{ jj}" format.name$ 
         "\onerabstbox[others]" =
            { " " * "et~al." * }
            { word.and * editor #2 "{vv~}\onerabstbox[{ll}]" format.name$ * }
          if$
        }
      if$
    }
  if$
}

FUNCTION {format.book.crossref}
{ volume empty$
    { "volume vide dans " cite$ * "'s crossref of " * crossref * warning$
      word.in
    }
    { "Vol." volume tie.or.space.connect
      " \ofname{} " *
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { series empty$
            { "Il faut editor, key, ou series pour " cite$ * 
              " pour r\'ef crois\'ee " * crossref * warning$
              "" *
            }
            { series emphasize * }
          if$
        }
        { key * }
      if$
    }
    { format.crossref.editor * }
  if$
  "\cite{" * crossref * "}" *
}

FUNCTION {format.incoll.inproc.crossref}
{
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { booktitle empty$
            { "Il faut editor, key, or booktitle pour " cite$ * 
              " pour r\'ef crois\'ee " * crossref * warning$
              ""
            }
            { word.in booktitle emphasize * }
          if$
        }
        { word.in key * " " *}
      if$
    }
    { word.in format.crossref.editor * " " *}
  if$
  "\cite{" * crossref * "}" *
}

FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    { journal "journal" output.check
      format.vol.num output
	  format.pages output
      format.date "year" output.check
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  new.block
  format.btitle "title" output.check
  new.block     % Ajoute le 10/12/99 version 3, precedemment efface par erreur?
  crossref missing$
    { format.bvolume output
      %new.block                        % Enleve le 10/12/99 version 3
      format.number.series output
      new.sentence
      publisher "publisher" output.check
      address output
    }
    {
      %new.block                        % Enleve le 10/12/99 version 3
      format.book.crossref output.nonnull
    }
  if$
  format.edition output
  format.date "year" output.check
  new.block
  note output
  fin.entry
}

FUNCTION {booklet}
{ output.bibitem
  format.authors output
  new.block
  format.title "title" output.check
  new.block
  howpublished output
  address output
  format.date output
  new.block
  note output
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  new.block
  format.btitle "title" output.check
  new.block  %  Ajoute le 10/12/99  Version 3
  crossref missing$
    { format.bvolume output
      format.chapter.pages "chapter and pages" output.check
      % new.block  %  Enleve le 10/12/99  Version 3
      format.number.series output
      new.sentence
      publisher "publisher" output.check
      address output
    }
    { format.chapter.pages "chapter and pages" output.check
      % new.block  %  Enleve le 10/12/99  Version 3
      format.book.crossref output.nonnull
    }
  if$
  format.edition output
  format.date "year" output.check
  new.block
  note output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.bvolume output
      format.number.series output
      format.chapter.pages output
      new.sentence
      publisher "publisher" output.check
      address output
      format.edition output
      format.date "year" output.check
    }
    { format.incoll.inproc.crossref output.nonnull
      format.chapter.pages output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.bvolume output
      format.number.series output
      format.pages output
      address empty$
        { organization publisher new.sentence.checkb
          organization output
          publisher output
          format.date "year" output.check
        }
        { address output.nonnull
          organization output  % passe avant la date version 3.0
          publisher output     % passe avant la date version 3.0
          format.date "year" output.check
%          new.sentence       % enleve version 3.0
        }
      if$
    }
    { format.incoll.inproc.crossref output.nonnull
      format.pages output
      format.date "year" output.check
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {conference} { inproceedings }

FUNCTION {manual}
{ output.bibitem
  author empty$
    { organization empty$
        'skip$
        { organization output.nonnull
          address output
        }
      if$
    }
    { format.authors output.nonnull }
  if$
  new.block
  format.btitle "title" output.check
  author empty$
    { organization empty$
    {
          address new.block.checka
          address output
        }
        'skip$
      if$
    }
    {
      organization address new.block.checkb
      organization output
      address output
    }
  if$
  format.edition output
  format.date output
  new.block
  note output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.btitle "title" output.check
  new.block
  "\masterthesisname{}" format.thesis.type output.nonnull
  school "school" output.check
  address output
  format.date "year" output.check
  new.block
  note output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors output
  title howpublished new.block.checkb
  format.title output
  howpublished new.block.checka
  howpublished output
  format.date output
  new.block
  note output
  fin.entry
}

FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.btitle "title" output.check
  new.block
  "\phdthesisname{}" format.thesis.type output.nonnull
  school "school" output.check
  address output
  format.date "year" output.check
  new.block
  note output
  fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
  editor empty$
    { organization output }
    { format.editors output.nonnull }
  if$
  new.block
  format.btitle "title" output.check
  new.block
  format.bvolume output
  format.number.series output
  address empty$
    { editor empty$
        { publisher new.sentence.checka }
        { organization publisher new.sentence.checkb
          organization output
        }
      if$
      publisher output
      format.date "year" output.check
    }
    { address output.nonnull
      format.date "year" output.check
      new.sentence
      editor empty$
        'skip$
        { organization output }
      if$
      publisher output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  format.tr.number output.nonnull
  address output
  format.date "year" output.check
  new.block
  note output
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  note "note" output.check   % Car le champ note est requis pour unpublished
  fin.entry
}

FUNCTION {default.type} { misc }

MACRO {jan} {"\janname{}"}
MACRO {feb} {"\febname{}"}
MACRO {mar} {"\marname{}"}
MACRO {apr} {"\aprname{}"}
MACRO {may} {"\mayname{}"}
MACRO {jun} {"\junname{}"}
MACRO {jul} {"\julname{}"}
MACRO {aug} {"\augname{}"}
MACRO {sep} {"\sepname{}"}
MACRO {oct} {"\octname{}"}
MACRO {nov} {"\novname{}"}
MACRO {dec} {"\decname{}"}

READ

STRINGS { longest.label }

INTEGERS { number.label longest.label.width }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}

FUNCTION {longest.label.pass}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}

EXECUTE {initialize.longest.label}

ITERATE {longest.label.pass}

FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  bstpreamble empty$
    'skip$
    { bstpreamble write$ newline$ }
  if$
  "\begin{thebibliography}{"  longest.label  * "}" * write$ newline$
}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
  nb.refs int.to.str$ " refs" *
  " (" version * ")." *
  * top$ 
}

EXECUTE {end.bib}
