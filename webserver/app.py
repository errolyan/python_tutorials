from _init_ import app
from asr import google_asr
from flask import jsonify, render_template, request
import util


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/test/index')
def index():
    return render_template("index.html")


@app.route('/analyse', methods=['POST'])
def analyse():
    country = request.args.get('country', None)
    platform = request.args.get('platform', None)
    tar_lan_code = request.args.get('tar_lan_code', None)
    product_id = request.args.get('product_id', None)
    device_mac = request.args.get('device_mac', None)

    text = request.form.getlist('text')[0]

    if country is not None and platform is not None and tar_lan_code is not None and product_id is not None \
            and device_mac is not None:
        util.recordlog_to_bigdata(country, tar_lan_code, platform, product_id, device_mac, text, None, "analyse")
    return util.nlp_analyze(text)


# support for binary data
@app.route('/google/asr', methods=['POST'])
def asr():
    req_audio_bin_data = request.get_data()
    audio_path = request.args.get('audio_path', None)
    country = request.args.get('country', None)
    platform = request.args.get('platform', None)
    tar_lan_code = request.args.get('tar_lan_code', None)
    product_id = request.args.get('product_id', None)
    device_mac = request.args.get('device_mac', None)

    if req_audio_bin_data or audio_path:  # req_audio_bin_data preferred
        unqGoogleASRHere = None
        if not google_asr.unqGoogleASR:
            unqGoogleASRHere = google_asr.initGoogleASR()  # 初始化GoogleASR客户端服务(单例)
        else:
            unqGoogleASRHere = google_asr.unqGoogleASR

        transcribed, auxiliary_info = unqGoogleASRHere.parse_n_response(
            audio_path, tar_lan_code, req_audio_bin_data)

        if tar_lan_code is not None:
            util.record_corpus(tar_lan_code, req_audio_bin_data)
        if country is not None and platform is not None and tar_lan_code is not None and product_id is not None \
                and device_mac is not None:
            util.recordlog_to_bigdata(country, tar_lan_code, platform, product_id,
                                  device_mac, transcribed, req_audio_bin_data, "/google/asr")
        return jsonify({'transcribed': transcribed, 'auxiliary_info': auxiliary_info, 'product_id': product_id,
                        'device_mac': device_mac})
    else:
        return jsonify({'auxiliary_info': 'invalid req_audio_bin_data or audio_path detected'})


# combine asr with nlp
@app.route('/google/nlp', methods=['POST'])
def asr_nlp():
    asr_result = asr()

    if asr_result and asr_result.json:
        asr_result = asr_result.json
        if 'transcribed' in asr_result and asr_result['transcribed']:
            return util.nlp_analyze(asr_result['transcribed'])
        return jsonify({'auxiliary_info': 'failed to transcribe the source'})
    else:
        return jsonify({'auxiliary_info': 'invalid asr_result or asr_result.json detected due to asr failure'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # google_asr.initGoogleASR()  # 初始化GoogleASR客户端服务(单例)
