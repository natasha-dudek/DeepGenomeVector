def confirm_ribosome(gen_kos):
    """Evaluates how many genes encoding the ribosome are present in an input
    genome vector.

    Arguments:
            gen_kos (list) -- KO numbers encoded by genome vector
    """
    small_subunit = {
        "K02945": "S1",
        "K02967": "S2",
        "K02982": "S3",
        "K02986": "S4",
        "K02988": "S5",
        "K02990": "S6",
        "K02992": "S7",
        "K02994": "S8",
        "K02996": "S9",
        "K02946": "S10",
        "K02948": "S11",
        "K02950": "S12",
        "K02952": "S13",
        "K02954": "S14",
        "K02956": "S15",
        "K02959": "S16",
        "K02961": "S17",
        "K02963": "S18",
        "K02965": "S19",
        "K02968": "S20",
        "K02970": "S21",
        "K19032": "S30",
        "K19033": "S31",
    }
    large_subunit = {
        "K02863": "L1",
        "K02886": "L2",
        "K02906": "L3",
        "K02926": "L4",
        "K02931": "L5",
        "K02933": "L6",
        "K02935": "L7/L12",
        "K02939": "L9",
        "K02864": "L10",
        "K02867": "L11",
        "K02871": "L13",
        "K02874": "L14",
        "K02876": "L15",
        "K02878": "L16",
        "K02879": "L17",
        "K02881": "L18",
        "K02884": "L19",
        "K02887": "L20",
        "K02888": "L21",
        "K02890": "L22",
        "K02892": "L23",
        "K02895": "L24",
        "K02897": "L25",
        "K02899": "L27",
        "K02902": "L28",
        "K02904": "L29",
        "K02907": "L30",
        "K02909": "L31",
        "K02911": "L32",
        "K02913": "L33",
        "K02914": "L34",
        "K02916": "L35",
        "K02919": "L36",
        "K07590": "L7A",
    }

    def confirm_or_deny(gen_kos, subunit_dict):
        count = 0
        for key in subunit_dict:
            try:
                assert key in gen_kos
                count += 1
            except AssertionError:
                print("missing RP", subunit_dict[key])
        return count

    ssu_count = confirm_or_deny(gen_kos, small_subunit)
    lsu_count = confirm_or_deny(gen_kos, large_subunit)
    print("ssu_count", ssu_count, "/", len(small_subunit))
    print("lsu_count", lsu_count, "/", len(large_subunit))


def confirm_rrna(gen_kos):
    rrna = {"K01980": "23S rRNA", "K01985": "5S rRNA", "K01977": "16S rRNA"}
    count = 0
    for key in rrna:
        try:
            assert key in gen_kos  # 23S ribosomal RNA
            count += 1
        except:
            print("missing", rrna[key])
    print("present", count, "/", len(rrna))


def confirm_51_bscgs(gen_kos):
    """Evaluates how many rRNA genes are encoded by an input genome vector.

    Arguments:
            gen_kos (list) -- KO numbers encoded by genome vector
    """
    bscgs = {
        "K01872": "alanyl tRNA synthetase",
        "K01887": "arginyl tRNA synthetase",
        "K22503": "aspartyl tRNA synthetase",
        "K02469": "gyrA",
        "K01892": "Histidyl tRNA synthetase",
        "K01869": "leucyl tRNA synthetase",
        "K01889": "Phenylalanyl tRNA synthetase alpha",
        "K03076": "Preprotein translocase subunit SecY",
        "K03553": "recA",
        "K02863": "L1",
        "K02864": "L10",
        "K02867": "L11",
        "K02871": "L13",
        "K02874": "L14",
        "K02876": "L15",
        "K02878": "L16 L10E",
        "K02879": "L17",
        "K02881": "L18",
        "K02884": "L19",
        "K02886": "L2",
        "K02887": "L20",
        "K02888": "L21",
        "K02890": "L22",
        "K02892": "L23",
        "K02895": "L24",
        "K02899": "L27",
        "K02904": "L29",
        "K02906": "L3",
        "K02907": "L30",
        "K02926": "L4",
        "K02931": "L5",
        "K02933": "L6P L9E",
        "K02946": "S10",
        "K02948": "S11",
        "K02950": "S12",
        "K02952": "S13",
        "K02956": "S15",
        "K02959": "S16",
        "K02961": "S17",
        "K02963": "S18",
        "K02965": "S19",
        "K02967": "S2",
        "K02968": "S20",
        "K02982": "S3",
        "K02986": "S4",
        "K02988": "S5",
        "K02990": "S6",
        "K02992": "S7",
        "K02994": "S8",
        "K02996": "S9",
        "K01873": "Valyl tRNA synthetase",
    }
    count = 0
    for key in bscgs:
        try:
            assert key in gen_kos
            count += 1
        except:
            print("missing", bscgs[key])
    print("present", count, "/", len(bscgs))
