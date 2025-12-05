# report_project
課題共有用に作成

課題設定:<br>
https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py<br>
の元コードを対象に「uvによるパッケージ管理」「型ヒントの追加」「pytestによるテスト追加」を行った。<br>

作業内容:<br>
・元コードではrequirements.txtで管理されていたパッケージ管理をuvによるパッケージ管理に変更<br>
・元コードに型ヒントを追加<br>
・元コードを対象としたpytestによるテストコード「tests/test_swinunet.py」を追加<br>
・pytestによるテストコードを実行したときに出る warning を解消するために元コードを修正<br>

ファイルの説明:<br>
・「src/swin_transformer_unet_skip_expand_decoder_sys.py」 ・・・  元コード https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py の修正後コード<br>
・「tests/test_swinunet.py」 ・・・  pytestによるテストコード<br>