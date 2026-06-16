\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{stageOnera}[2015/11/18 Classe ONERA JDD]

%% Chargement de la classe mère
\LoadClass[a4paper,11pt]{article}

%% Chargement de paquets
\RequirePackage{graphicx}
\RequirePackage{multirow}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage[hmargin=1.5cm,vmargin=2cm]{geometry}
\RequirePackage{ragged2e}
\RequirePackage{eso-pic}
\RequirePackage{times}
\RequirePackage{sectsty}
\RequirePackage{microtype}
\RequirePackage{cellspace}


%% Déclaration des variables

\let\@nom\@empty
\let\@prenom\@empty
\let\@titre\@empty
\let\@thematique\@empty
\let\@encadrantOnera\@empty
\let\@departement\@empty
\let\@formation\@empty
\let\@motsCles\@empty

%% Affectation des variables depuis l'instance de la classe
\newcommand{\nom}[1]{\gdef\@nom{#1}}
\newcommand{\prenom}[1]{\gdef\@prenom{#1}}
\newcommand{\titre}[1]{\gdef\@titre{#1}}
\newcommand{\thematique}[1]{\gdef\@thematique{#1}}
\newcommand{\encadrantOnera}[1]{\gdef\@encadrantOnera{#1}}
\newcommand{\departement}[1]{\gdef\@departement{#1}}
\newcommand{\formation}[1]{\gdef\@formation{#1}}
\newcommand{\motsCles}[1]{\gdef\@motsCles{#1}}

%% Logo Onera
\newcommand\AtLowerRightCorner[3]{%
  \put(\LenToUnit{\dimexpr\paperwidth-#1},\LenToUnit{#2}){#3}%
}
% ajout de texte ou d'images en haut à gauche, en haut à droite, etc.
\AddToShipoutPicture{%
\AtLowerRightCorner{6cm}{0.5cm}{\includegraphics[width=144.54pt]{figures/logos/ONERA_logo.pdf}}% image en bas à droite
}

%% Titre
\def\tabularxcolumn#1{m{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}m{#1}}
\newcolumntype{K}{>{\centering\arraybackslash}X}
\newcolumntype{M}{>{\raggedright\arraybackslash}X}

\renewcommand\maketitle{
%%% Bloc titre
\renewcommand{\arraystretch}{1.5}
\begin{center}
  %\begin{tabularx}{\textwidth}{|C{4cm}|K|}
 % \begin{tabularx}{\textwidth}{|K|}  
  %\begin{tabularx}{\textwidth}{|l M|}

 \begin{tabularx}{\textwidth}{|l M|} 
     \hline
 %   \vspace{10pt}& \textbf{\textit{\LARGE{\@titre}}}  \\[5pt] 
 %   \vspace{10pt}
 \multicolumn{2}{|c|}{\textbf{\textit{\LARGE{\@titre}}}}\\
%   \vspace{10pt}
%    \multirow{7}{*}{
%      \vspace{10pt}
%      \centering
%      \includegraphics[width=3cm,height=4cm,keepaspectratio]{\@photo}
%      }
   %  \vspace{10pt} \textbf{\textit{\LARGE{\@titre}}} & \\[5pt] 

   \textbf{Nom : \Large{\@nom}}& \\
   \textbf{Prénom : \large{\@prenom}}& \\
    Département/Unité : \@departement &\\
    \textbf{Thematique} \@thematique &\\
   \textbf{Encadrant(s) ONERA} & \@encadrantOnera \\
   %(nom, pr\'e{}nom, d\'e{}pt/unit\'e{})
    
    \textbf{Formation} & \@formation \\
    \hline
  \end{tabularx}
  
\end{center}

%% Bloc des encadrants
%\renewcommand{\arraystretch}{1.3}
%\begin{center}
%  \begin{tabularx}{\textwidth}{|lM|}
%    \hline
%    \textbf{Directeur(s) de th\`e{}se} (nom, pr\'e{}nom, institution)
%    & \@directeurThese \\
%    \textbf{Encadrant(s) ONERA} (nom, pr\'e{}nom, d\'e{}pt/unit\'e{})
%    & \@encadrantOnera \\
%    \textbf{Site}
%    & \@site \\
% 
%    \hline
%  \end{tabularx}
%\end{center}
\renewcommand{\arraystretch}{1}

\noindent\textbf{Mots cl\'e{}s :} \@motsCles

}

%% Taille des sections
\sectionfont{\fontsize{11}{15}\selectfont}

%% Suppression de la numérotation de page
\thispagestyle{empty}
\pagestyle{empty}

\endinput
