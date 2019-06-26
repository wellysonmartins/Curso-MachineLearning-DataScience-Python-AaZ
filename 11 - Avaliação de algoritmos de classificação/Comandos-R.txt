MAIS INFORMAÇÕES

http://kourentzes.com/forecasting/2014/04/19/tstools-for-r/

INSTALAÇÃO

if (!require("devtools"))
    install.packages("devtools")
devtools::install_github("trnnick/TStools")

require(TStools)

dados <- read.csv("<path do arquivo>")
matriz <- as.matrix(dados)
TStools::nemenyi(matriz,conf.int=0.95, plottype="vline")