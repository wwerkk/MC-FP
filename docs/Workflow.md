```plantuml
object Notebook
Notebook : local or remote
Notebook : - data loading
Notebook : - data preprocessing
Notebook : - model training
Notebook : - sequence generation (offline)
Notebook : - signal resynthesis (offline)

object Script
Script : local 
Script : - preprocessed data loading
Script : - sequence generation (online)
Script : - UDP communication

object Max
Max : local
Max : - buffer lookup
Max : - signal resynthesis (online)
Max : - UDP communication


Notebook --> "(little) data" Script
Notebook --> "model" Script

Script --> "generated tokens" Max

```
