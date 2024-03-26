function reverse_complement(seq){
    let dst = "";
    const map = new Map();
    var n = "";
    map.set('A','T');
    map.set('T','A');
    map.set('C','G');
    map.set('G','C');
    for (let i = seq.length-1;i>-1;i--){
        n = map.get(seq[i]);
        if (n != undefined){
            dst += n;
        }
    }
    return dst;
};

function sanitize(seq){
    let dst = "";
    const map = new Map();
    var n = "";
    map.set('A','T');
    map.set('T','A');
    map.set('C','G');
    map.set('G','C');
    for (let i = 0;i<seq.length;i++){
        n = map.get(seq[i]);
        if (n != undefined){
            dst += seq[i];
        }
    }
    return dst;
}


function translate(s){
    let seq = sanitize(s)
    const DNA_Codons = {
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "TGT": "C", "TGC": "C",
        "GAT": "D", "GAC": "D",
        "GAA": "E", "GAG": "E",
        "TTT": "F", "TTC": "F",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
        "CAT": "H", "CAC": "H",
        "ATA": "I", "ATT": "I", "ATC": "I",
        "AAA": "K", "AAG": "K",
        "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "ATG": "M",
        "AAT": "N", "AAC": "N",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAA": "Q", "CAG": "Q",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "TGG": "W",
        "TAT": "Y", "TAC": "Y",
        "TAA": "_", "TAG": "_", "TGA": "_"
    }
    const numCodons = (seq.length - seq.length%3)/3;
    let protein = "";
    for (let i = 0;i < numCodons;i ++){
        protein += DNA_Codons[seq[3*i] + seq[3*i+1] + seq[3*i+2]]
    }
    return protein;
}

function orf(seq){
    const seq_r = reverse_complement(seq);
    var orfs = []
    var orf_candidates = "";
    for (let i = 0;i<seq.length;i++){
        if (seq.slice(i,i+3) == "ATG"){
            orf_candidates = translate(seq.slice(i,seq.length)).split("_")
            if (orf_candidates.length>1){
                orfs.push(orf_candidates[0]);
            }
        }
        if (seq_r.slice(i,i+3) == "ATG"){
            orf_candidates = translate(seq_r.slice(i,seq_r.length)).split("_")
            if (orf_candidates.length>1){
                orfs.push(orf_candidates[0]);
            }
        }
    }
    return Array.from(new Set(orfs));
}

const seq = "CGCACTGAGATATGAAAGAAACACGACCTGCTATCCTGTTCGCGTGCTGCTTGTCGTTAGTTAAGAGGGGCGTTCAAGAAACGACAATGGCTTACTGTCCAGGATAGAACCGAAAGAGATCGTGATAACGACGTCCCACTCGCCTAAATTTCAACGGAGGGCTTTAAGATGGAGAACGACGACCTTCCGCGGAGCAGTACGGTAACGCTAATAACATAGCCTCTAGGTTTTGAAGCCAGTAATGCTGAGCGCCGTAGCGATTGGTCCTGGTCCCAGATTCGGCTATCATCTATGTCTTCATATGCCCTCCTCACACGCGGGAATATAGGTACTTTTAAGTAATCAGGAGAGACAATGACGCCTCCTTATACCGTTGTTGATGGAGTTCATGTAAGGAATCAACTGACGCATAGAGGTCCGGGCCAGAGCGCAGGAACAAATGGCACGAGTAAAAGATACACGCTTAACTTAGCTAAGTTAAGCGTGTATCTTTTACTCGTGCCATATCAACTTCTGCGTCCTAGGCCCAACCCTTTAAAAAGTCGAGATAACGTTGAGACTAGTAAAGACTACCGCGTTGTATATGTAAGAGCACTCGGACGACGAGTTCTGAGTGTTGATATAGAGCGTTGTTGCGCTTGTGTCGCTCGAACCGTAAGATGAGTCTAACGGCGCGGTCCGATCGGAACTTGCTTATGAGCCTTCTAGACGATCACTCGATGGCGGTTAATATGTAAACCAAGGGAGGCCATATGGTTTATTCCTTCAGTGGCGCTGCTCCATAATGCTTAGGTATTGGTCCTCTAGCATTCCAGTCACGCGCCATCTGTCTGCGGATGAGGGTCGACGCCGCGAAGCAGGGTGTACAGAACTGACAACGCGCACTATTTGCACTTATAAAATGCGCAAACAGCCCCCGACTTAACTACGTTACCGACGTGGAT"
const result = orf(seq);
for (let i = 0;i<result.length;i++){
    console.log(result[i]);
}