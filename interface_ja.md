# 所有権の原則

ExoFamily全体で次のルールに統一する


## 原則A：providerは自分の物理量を一般的な形で公開する
ExoEOSはEOS state、density、fugacity、activityを公開
ExoGibbsはspecies、mole fraction、equilibrium stateを公開
ExoJAXはpressure grid、大気構造、opacity、RTを公開

providerは、特定の利用先を知らないようにします。

例えばExoEOSはExoGibbsやExoJAXをimportしません。

## 原則B：consumerがportを所有する
ExoGibbsが必要とするfugacity callbackはExoGibbsが定義
ExoJAXが必要とするdensity providerやcomposition inputはExoJAXが定義
ExoJAXが必要とするpressure boundary contractはExoJAXが定義
原則C：pair-specific adapterはconsumer側の interop に一つだけ置く

### 命名の統一

<consumer>.interop.<provider>

とします。

exogibbs.interop.exoeos
exojax.interop.exogibbs
exojax.interop.exoeos

ただし、単純なarray contractで直接接続できる場合には、pair-specific adapter自体を作らない方がよいです。

## 原則D：core moduleはsibling packageをimportしない

例えば、

import exogibbs

でExoEOSが必要になってはいけません。

実際にadapterを使用するときだけ、

from exogibbs.interop.exoeos import make_pure_lnphi_func

をimportします。

interop/__init__.py は空またはlazyにし、top-level importから他packageを読み込ませません。