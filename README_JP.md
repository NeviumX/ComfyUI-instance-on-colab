[English](README.md) | 日本語

# ComfyUI instance on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeviumX/ComfyUI-instance-on-colab/blob/main/ComfyUI_instance_on_Colab_JP.ipynb)

Google ColaboratoryのGPUでComfyUIを実行し、ブラウザーやComfy Desktopの **Remote Connection** から使うためのノートブックです。

- Googleドライブに置いたモデルを使えます。Hugging Face・CivitaiからGoogleドライブへダウンロードもできます。
- ComfyUIの環境・カスタムノード・設定をGoogleドライブへ保存し、次のランタイムで復元します。
- 接続方法を3つから選べます。プライベートURL、ペアリングコード付きの公開URL、独自ドメインのURLです。

## 必要なもの

- GPUを利用できるGoogle Colab。有料プランと計算ユニット残高を前提とします。
- モデルと環境を保存できるGoogleドライブ、およびモデル全体をコピーできるColabの空き容量。
- Remote Connectionを使う場合は [Comfy Desktop](https://github.com/Comfy-Org/Comfy-Desktop)。使わない場合は通常のブラウザーだけで利用できます。

## 使い方

1. 上の **Open in Colab** をクリックします（日本語版のノートブックが開きます。英語版は [英語のREADME](README.md) のボタンから開けます）。[ComfyUI_instance_on_Colab_JP.ipynb](ComfyUI_instance_on_Colab_JP.ipynb) または [ComfyUI_instance_on_Colab_EN.ipynb](ComfyUI_instance_on_Colab_EN.ipynb) をダウンロードして、[Google Colab](https://colab.research.google.com/) へアップロードすることもできます。内容は同じで、説明文とメッセージの言語だけが異なります。
2. GPUランタイムを選び、1〜3のセル（設定・コードの展開・セットアップ）を実行します。表示されるGoogleの画面でドライブへの接続を許可します。各セルは入力欄とチェックボックスで操作し、チェックボックスのある操作はONにして実行したときだけ行われます。
3. Googleドライブの `ComfyUI-Colab/models/` にある種類別のフォルダーへモデルを置いて5を実行するか、Hugging Face・CivitaiのURLを4へ貼り付けます。
4. 6-1〜6-3のいずれかを実行してComfyUIを起動し、表示されたURLを開きます（下の「接続方法」）。
5. 終了するときは、PCへの出力の取得を確認してから10を実行し、完了を待ってからランタイムを削除します。

## 接続方法

| セル | 使う場面 | 接続できる人 |
|---|---|---|
| **6-1 プライベートURL** | ノートブックを開いているブラウザーで使う | ノートブックを実行しているGoogleアカウントでログインしたブラウザー。Comfy Desktopからは使えません。 |
| **6-2 公開URL** | Comfy Desktopや別の端末から使う | URLを開き、ノートブックに表示されたペアリングコードを入力した端末 |
| **6-3 独自ドメインURL** | 自分のドメインの固定URLで使う | Cloudflare Accessでログインした、許可済みのアカウント |

- 別の方法へ切り替えるときは、先に10で停止します（`FLUSH_AND_UNMOUNT` はOFF）。
- **6-2:** ペアリングコードは10分間・1回だけ有効です。端末を追加するときは、`NEW_PAIRING_CODE` をONにして6-2を実行します。停止すると、URLと接続済みの端末はすべて無効になります。
- **Comfy Desktop:** URLをRemote Connectionに入力し、**Auto-download outputs** をONにすると生成物がPCへ保存されます。
- **停止ボタン:** ComfyUIの画面上部の **停止** から、環境を保存して接続を解除できます。そのままランタイムを削除することも選べます。削除すると、PCへ取得していない出力は失われます。

### 独自ドメインURL（6-3）の準備

Cloudflareのアカウント、CloudflareのDNSで管理しているドメイン、Cloudflare Zero Trustが必要です。準備は最初の1回だけです。Cloudflareのダッシュボードのメニュー名は多少異なる場合があります。

1. Zero Trustで、使用するホスト名の **Accessアプリケーション**（セルフホスト）を作成し、自分のメールアドレスだけを許可するポリシーを付けます。ログイン方法を選びます。メールのワンタイムPINは、Comfy Desktopの中でも使えます。トンネルより先に作成してください。
2. **トンネル**（コネクターの種類はcloudflared）を作成し、トークン（`eyJ` で始まる長い文字列）をコピーします。コネクターをどこかへインストールする必要はありません。トンネルに公開ホスト名を追加します。ホスト名は自分のもの、サービスの種類はHTTP、URLは `127.0.0.1:8189` です。
3. Accessアプリケーションの **Application Audience (AUD) Tag** をコピーします。
4. Colabのシークレット（鍵アイコン）に、`CF_TUNNEL_TOKEN` の名前でトークン、`CF_ACCESS_AUD` の名前でAUDタグを登録し、このノートブックからのアクセスを許可します。
5. 6-3にホスト名を入力して実行します。表示されたURLを開いてログインします。

- トンネルのトークンは秘密にしてください。Colabのシークレットにだけ保存し、漏れた可能性がある場合はCloudflareでトンネルを削除してください。
- 6-3は、ホスト名がAccessのログインを求め、トンネルの転送先が `127.0.0.1:8189` になっているときだけ接続します。Cloudflareの設定を変えた直後は、数分待ってから実行し直してください。

## 注意点

### セキュリティ

- Googleドライブへの接続は、ドライブ全体への許可になります。個人データを置かない専用のGoogleアカウントを使ってください。
- カスタムノードはランタイム上で独自のコードを実行します。信頼できるカスタムノードとモデルだけを導入してください。ノートブックはドライブと接続用の秘密情報をカスタムノードから保護しますが、完全な隔離ではありません。
- Googleドライブに保存した環境は次のランタイムで復元され、含まれるカスタムノードも再び実行されます。セットアップ時に名前を表示するので、不審なものがあれば1の `RESTORE_SAVED_ENVIRONMENT` をOFFにして新しい環境で始めてください。
- ペアリングコードとトークンは共有せず、ノートブックを共有する前にセルの出力を消去してください。ダウンロード用のトークンは、必要な場合だけColabのシークレットへ登録します。

### 保存

- 環境はComfyUIの実行中に自動でGoogleドライブへ保存されますが、突然の切断の直前に行った変更は復元されないことがあります。すぐに保存したいときは8を使います。
- 入力・出力ファイルはGoogleドライブへ保存しません。Desktopのプレビューと、Desktopが接続していない間に生成した出力は自動では取得されません。9でZIPにまとめて回収できます。

### ComfyUIの更新

- 保存した環境を復元すると、ComfyUIは保存したときのバージョンのままです。
- カスタムノード・設定・ワークフローを残したままComfyUIを更新するには、0をONにして実行（最新リリースを選びます）するか、1の `COMFY_REF` に更新先を入力して、1 → 2 → 3 → 11 → 6-1〜6-3のいずれか の順に実行します。起動中の場合は、先に10で停止します（`FLUSH_AND_UNMOUNT` はOFF）。
- 更新に失敗した場合は元のバージョンへ戻し、Googleドライブの保存内容は変えません。
- このノートブックは `COMFY_REF` の既定値で動作を確認しています。

### 画面が表示されないとき

- 初回の表示には、回線によって1分前後かかります。
- ロゴのまま進まない場合は、7の `RUN_CONNECTION_DIAGNOSIS` をONにして実行します。接続が止まっている箇所を表示します。問題が見つからない場合は、PC側（広告ブロック・セキュリティソフト・ブラウザー拡張）を確認してください。
- 6-1では、ブラウザーがノートブックと同じGoogleアカウントでログインしているかを確認してください。

### 利用条件

- [Colabの利用条件](https://research.google.com/colaboratory/faq.html) と [Quick Tunnelの制限](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) に従って使用してください。
