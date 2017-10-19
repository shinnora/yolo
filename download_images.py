import urllib.request

items = [
    ["http://3.bp.blogspot.com/-Pio2FOWygvY/UqmPJINDZ5I/AAAAAAAAbeA/onvHuySB4Kc/s800/animal_kowai_kuma.png", "./items/kuma.png"],
    ["http://4.bp.blogspot.com/-WBhyOXH1-wY/UaVVKj7uOOI/AAAAAAAAUA4/d0AWj2TV880/s800/animal_uma_ninjin.png", "./items/uma.png"],
    ["http://4.bp.blogspot.com/-E6neBKIc2Og/UQuSVS5jvdI/AAAAAAAALvQ/zMZMO8lZ9Ds/s1600/shoes_41.png", "./items/alpaca.png"],
    ["http://1.bp.blogspot.com/-aas6GmfUCwo/Vn-CoeFVUdI/AAAAAAAA2MA/2LPeV_t4ziw/s800/eto_remake_nezumi.png", "./items/nezumi.png"],
    ["http://4.bp.blogspot.com/-OvqTTiAK1sc/Vn-CpKKjrsI/AAAAAAAA2MY/Y_IAu9aeGMc/s800/eto_remake_ushi.png", "./items/ushi.png"],
    ["http://3.bp.blogspot.com/-0NH6yQqogJw/Vn-Cns03meI/AAAAAAAA2L0/G6A4S-8YTpc/s800/eto_remake_inu.png", "./items/inu.png"],
    ["http://4.bp.blogspot.com/-EZwH6YbHgSc/Vn-Co3KQc3I/AAAAAAAA2MM/sFjbcDsMt8A/s800/eto_remake_tori.png", "./items/tori.png"],
    ["http://2.bp.blogspot.com/-oIWwEwe664Q/Vn-CoheC78I/AAAAAAAA2ME/aQ7RU4zPNbE/s800/eto_remake_tora.png", "./items/tora.png"],
    ["http://4.bp.blogspot.com/-DoFXrGF5DNU/Vn-CpGg4ahI/AAAAAAAA2MU/HP-y_XxtvSw/s800/eto_remake_usagi.png", "./items/usagi.png"],
    ["http://4.bp.blogspot.com/-Lkh2UBHwMZw/VjK0WwSa5AI/AAAAAAAA0Ow/hQJcjZqeeFE/s800/eto_saru_fukubukuro.png", "./items/saru.png"]
]

backgrounds = [
    ["https://www.pakutaso.com/shared/img/thumb/TSURU170321b-35%20mm-014_TP_V.jpg", "./backgrounds/0.jpg"],
    ["https://www.pakutaso.com/shared/img/thumb/ELFADSC08970_TP_V.jpg", "./backgrounds/1.jpg"],
    ["https://www.pakutaso.com/shared/img/thumb/SORA0I9A4478_TP_V.jpg", "./backgrounds/2.jpg"],
    ["https://www.pakutaso.com/shared/img/thumb/ELFASKYDSC07129_TP_V.jpg", "./backgrounds/3.jpg"],
    ["https://www.pakutaso.com/shared/img/thumb/KUMO160219080I9A0119_TP_V.jpg", "./backgrounds/4.jpg"],
    ["https://www.pakutaso.com/shared/img/thumb/PPH_yuuyakezora_TP_V.jpg", "./backgrounds/5.jpg"],
    ["https://www.pakutaso.com/shared/img/thumb/M157_aozoratokumo_TP_V.jpg", "./backgrounds/6.jpg"],
]

for item in items:
    urllib.request.urlretrieve(item[0], item[1])

for background in backgrounds:
    urllib.request.urlretrieve(background[0], background[1])
