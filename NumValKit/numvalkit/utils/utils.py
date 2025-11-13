import os
import concurrent.futures
import pandas as pd

max_workers = 160

category_to_packages = {
    "off": ["*screen_off*"],
    "mobility": [
        "cn.ehi.carservice",
        "com.sdu.didi.hmos.psnger",
        "com.zuche.rentcar",
        "com.smartmetro.gzmetro",
        "com.baidu.robotaxi.huawei",
        "com.fawvw.jetta",
        "cn.volvo.vcdc",
        "com.lalamove.huolala.clienthar",
    ],
    "finance": [
        "com.alipay.mobile.client",
        "com.cmbchina.ccc.cmblife",
        "com.cmbchina.harmony",
        "com.huawei.hmos.quickpay",
        "com.huawei.hmos.wallet",
        "com.boc.bocsoft.mbs.personal",
        "cn.gov.tax.its",
        "com.unionpay.hmos.wallet",
        "com.huawei.hms.payment",
        "com.cmbchina.mbankhk",
        "com.cebbank.creditcard",
        "com.webank.wemoney",
        "com.dbs.sg.dbsmbanking",
        "com.icbc.elife",
        "com.duxiaoman.umoneypro",
        "com.hxb.mobile.hm",
        "com.ynrcc.ebank.hm",
        "com.gtja.junhong.hm.next",
    ],
    "travel": [
        "com.amap.hmapp",
        "com.mtr.mtrmobile",
        "com.tencent.map",
        "com.chinarailway.ticketingHM",
        "app.huawei.hmos.motor",
        "com.huawei.hms.floatingnavigation",
        "com.ctrip.harmonynext",
        "app.huawei.hmos.auto",
        "com.huawei.hmos.maps.app",
        "com.tmri.app.harmony12123",
        "com.klook.cn.hmos",
        "com.cbdn.zhoushan",
        "com.xiamenair.ohosapp",
        "com.csair.mbpohos",
        "com.youxiake",
    ],
    "system": [
        "com.android.permissioncontroller",
        "com.chuckfang.meow",
        "com.droi.tong",
        "com.huawei.hmos.ailife",
        "com.huawei.hmos.calendar",
        "com.huawei.hmos.calculator",
        "com.huawei.hmos.clock",
        "com.huawei.hmos.emergencycommunication",
        "com.huawei.hmos.finddevice",
        "com.huawei.hmos.hwstartupguide",
        "com.huawei.hmos.outerhome",
        "com.huawei.hmos.screenshot",
        "com.huawei.hmos.security.privacycenter",
        "com.huawei.hmos.settings",
        "com.huawei.hmos.tips",
        "com.huawei.hmos.vassistant",
        "com.huawei.hmsapp.appgallery",
        "com.huawei.hmsapp.himovie",
        "com.huawei.hmsapp.thememanager",
        "com.huawei.hmsapp.totemweather",
        "com.ohos.contacts",
        "com.ohos.mms",
        "com.ohos.notificationdialog",
        "com.ohos.sceneboard",
        "com.android.documentsui",
        "com.huawei.hmos.betaclub",
        "com.huawei.hmos.findservice",
        "com.huawei.hmos.slassistant",
        "com.huawei.hms.huaweiid",
        "com.huawei.hmsapp.intelligent",
        "com.huawei.hmsapp.samplemanagement",
        "com.huawei.hmos.instantshare",
        "com.huawei.hmos.ouc",
        "com.huawei.hmos.screenrecorder",
        "com.ohos.permissionmanager",
        "com.huawei.hmos.applock",
        "com.huawei.hmos.suggestion",
        "com.huawei.hmsapp.hisearch",
        "com.huawei.hmos.easymanage",
        "com.huawei.hmos.onekeylock",
        "com.huawei.hmos.hisuite",
        "com.huawei.hms.myfamily",
        "com.huawei.appmarket",
        "com.huawei.hms.textautofill",
        "com.huawei.hmos.spooler",
        "com.huawei.hmos.hiviewpage",
    ],
    "media": [
        "com.cctv.yangshipin.app.harmonyp",
        "com.hupu.heroes",
        "com.huawei.hmos.photos",
        "com.huawei.it.hmxinsheng",
        "com.ss.hm.article.news",
        "com.zhihu.hmos",
        "com.netease.news.hmos",
        "com.netease.party.huawei",
    ],
    "video": [
        "yylx.danmaku.bili",
        "com.huawei.hmsapp.himovie",
        "com.youku.next",
        "com.mxtech.videoplayer.ae",
        "com.modernsky.istv",
        "com.fivehundredpx.viewer.main",
    ],
    "music": [
        "com.tencent.hm.qqmusic",
        "com.huawei.hmsapp.music",
        "com.ximalaya.ting.xmharmony",
        "com.luna.hm.music",
        "com.imusic.tianlaiyunting",
        "com.tencent.karaokehm",
        "com.hmkuwo.cn",
    ],
    "browser": [
        "com.quark.ohosbrowser",
        "com.uc.mobile",
        "com.huawei.hmos.browser",
    ],
    "health": [
        "com.huawei.hmos.health",
    ],
    "productivity": [
        "com.huawei.hmos.calendar",
        "com.huawei.it.welink",
        "com.taobao.qqmail.hmos",
        "cn.wps.mobileoffice.hap",
        "cn.wps.mobileoffice.hap.ent.huawei",
        "com.tencent.qqmail.hmos",
        "com.huawei.hmos.email",
        "com.huawei.hmos.files",
        "com.huawei.hmos.notepad",
        "com.alicloud.hmdatabox",
        "com.baidu.netdisk.hmos",
        "com.huawei.hmos.meetimeservice",
        "com.huawei.hmos.superhub",
        "com.huawei.hmos.inputmethod",
        "com.huawei.hmos.hipreview",
        "com.ss.feishu",
        "app.xmind.cronut",
        "com.mubu.ohapp",
        "com.nowcoder.app",
        "com.github.kr328.clash.foss",
        "com.intsig.camscanner.hap",
        "com.mobisystems.office",
    ],
    "weather": ["com.huawei.hmsapp.totemweather"],
    "food_delivery": ["com.sdu.didi.hmos.psnger", "com.sankuai.hmeituan"],
    "communication": [
        "com.ohos.callui",
        "com.ohos.contacts",
        "com.ohos.mms",
        "com.tencent.mqq",
        "com.tencent.wechat",
        "com.huawei.hmos.meetime",
        "com.tencent.mm",
        "com.whatsapp.w4b",
        "com.ss.android.lark.seazen",
        "com.alibaba.android.rimet.gtmcoa",
    ],
    "shopping": [
        "com.jd.hm.mall",
        "com.taobao.idlefish4ohos",
        "com.taobao.taobao4hmos",
        "com.chinamobile.cmcc",
        "com.chinatelecom.esmarthome",
        "com.huawei.hmos.vmall",
        "cn.samsclub.hm.app",
        "com.cainiao.cainiao4hmos",
        "com.dewu.hos",
        "com.ingka.ikea.app.cn.hwoh.prod",
        "com.xunmeng.pinduoduo.hos",
        "com.jd.jdlite",
        "com.jd.pingou",
        "com.xiaomi.shop",
        "com.suning.ebuy.hos",
    ],
    "social_media": [
        "com.ss.hm.ugc.aweme",
        "com.xingin.xhs_hos",
        "com.immomo.momo",
        "com.ss.android.ugc.aweme.lite",
        "com.sina.weibo.stage",
        "com.kuaishou.hmapp",
        "com.soft.blued",
    ],
    "creativity": [
        "com.bytedance.dreamina",
        "com.deepseek.chat",
        "com.hos.moonshot.kimichat",
        "com.huawei.hmos.hiwrite",
    ],
    "reading": [
        "com.dragon.read.next",
        "com.huawei.hmsapp.books",
        "com.tencent.weread.hmos",
        "com.qidian.reader",
        "com.yuewen.qqreaderhm",
        "com.migu.cmread.hm",
        "com.shanbay.hmsentence",
        "huayang.com.myreader.wangyu.books",
    ],
    "gaming": [
        "com.huawei.hmsapp.gamecenter",
        "com.duole.hmos.zhuojimjhd.huawei",
        "com.huawei.hms.gameservice",
        "com.tencent.tmgp.sgame.hw",
        "com.zhuoyi.appstore.lite",
        "com.tencent.tmgp.sgamece.hw",
        "com.huawei.hmsapp.litegamelauncher",
        "com.huawei.hmsapp.litegames",
        "com.tencent.gamehelper.smoba",
        "com.tencent.tmgp.pubgmhd.hw",
        "com.netease.mc.huawei",
        "com.tuyoo.gomokupc.huawei.hm",
        "com.ztgame.bob.huaweihap",
        "com.tencent.ig",
        "com.igame.bjmf.yhsjmnq.huawei",
    ],
    "camera": [
        "com.huawei.hmos.camera",
        "com.huawei.camera",
        "com.jinrishuiyinxiangji.camera",
        "com.intsig.camscanner.hap",
    ],
    "tools": [
        "com.xiaomi.smarthome",
    ],
    "education": [
        "com.duolingo",
    ],
}

unknown_packages = set()
packages_to_category = {}
for category, packages in category_to_packages.items():
    for package in packages:
        packages_to_category[package] = category

# Apptype from IAWARE
type_names = [
    "system",  # TYPE_LAUNCHER = 1
    "communication",  # TYPE_SMS = 2
    "productivity",  # TYPE_EMAIL = 3
    "system",  # TYPE_INPUTMETHOD = 4
    "gaming",  # TYPE_GAME = 5
    "browser",  # TYPE_BROWSER = 6
    "reading",  # TYPE_EBOOK = 7
    "video",  # TYPE_VIDEO = 8
    "system",  # TYPE_SCRLOCK = 9
    "alarm",  # TYPE_ALARM = 10
    "communication",  # TYPE_IM = 11
    "music",  # TYPE_MUSIC = 12
    "travel",  # TYPE_NAVIGATION = 13
    "travel",  # TYPE_LOCATION_PROVIDER = 14
    "productivity",  # TYPE_OFFICE = 15
    "media",  # TYPE_GALLERY = 16
    "communication",  # TYPE_SIP = 17
    "social_media",  # TYPE_NEWS_CLIENT = 18
    "shopping",  # TYPE_SHOP = 19
    "system",  # TYPE_APP_MARKET = 20
    "tools",  # TYPE_LIFE_TOOL = 21
    "education",  # TYPE_EDUCATION = 22
    "finance",  # TYPE_MONEY = 23
    "camera",  # TYPE_CAMERA = 24
    "health",  # TYPE_PEDOMETER
    "mobility",  # rent car = 26
    "productivity",  # cloud storage = 27
    "business",  # driver,seller = 28
]
aosp_app_type_map = {}


def load_app_type(csv_path):
    global aosp_app_type_map
    df = pd.read_csv(csv_path)
    aosp_app_type_map = dict(zip(df["name"], df["type"]))


path = os.path.dirname(os.path.abspath(__file__))
load_app_type(f"{path}/app_type.csv")


def get_category_from_package(package_name):
    category = packages_to_category.get(package_name)
    if category:
        return category
    type_code = aosp_app_type_map.get(package_name)
    if type_code:
        if 1 <= type_code <= len(type_names):
            category = type_names[type_code - 1]
            # print(f"AOSP {package_name}, {category}")
            return category
        else:
            print(f"Warning: type code {type_code} for {package_name} out of range")
    unknown_packages.add(package_name)
    return "Unknown"


def get_app_from_package(package_name):
    if package_name == "com.ohos.sceneboard":
        return ""
    else:
        return package_name


def dict_to_dataframe(dict):
    key_set = category_to_packages.keys()
    result = [dict.get(key, 0) for key in key_set]
    return result


def get_overlap_seconds(start1, end1, start2, end2):
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)
    delta = (earliest_end - latest_start).total_seconds()
    return max(0, delta)  # 若无重叠，返回0


def dict_add(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in a.keys() | b.keys()}
