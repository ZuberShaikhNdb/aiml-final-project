# PaySim EDA
- Rows: 6,362,620; fraud: 8,213 (0.129%). **Heavy class imbalance**: accuracy would be 99.87% by predicting 'legit' always, so we use precision-recall AUC.
- Fraud only occurs in TRANSFER and CASH_OUT; within those types the fraud rate is 0.296% (2,770,409 rows). The model is trained and served on these two types.
- Median amount: fraud 441,423 vs legitimate 74,685.
- Origin accounts seen only once: 99.9%; destination accounts seen only once: 83.1%. Few repeated counterparties is why the graph model is evaluated on Elliptic instead.

| type | rows | fraud | fraud rate |
|---|---|---|---|
| CASH_IN | 1,399,284 | 0 | 0.0000% |
| CASH_OUT | 2,237,500 | 4,116 | 0.1840% |
| DEBIT | 41,432 | 0 | 0.0000% |
| PAYMENT | 2,151,495 | 0 | 0.0000% |
| TRANSFER | 532,909 | 4,097 | 0.7688% |