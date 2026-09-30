import os
import sys
import socket
import datetime
import ipaddress
import uvicorn

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def get_current_ip():
    """كشف الـ IP النشط حالياً سواء كان واي فاي أو نقطة اتصال هوتسبوت"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def generate_ssl_for_ip(ip_str):
    """توليد شهادة SSL محلية متوافقة 100% مع متطلبات آبل لنظام iOS"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cert_path = os.path.join(base_dir, "cert.pem")
    key_path = os.path.join(base_dir, "key.pem")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, ip_str),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u'Egyptian ALPR Scanner')
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    ip_obj = ipaddress.IPv4Address(ip_str)

    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.IPAddress(ip_obj),
                x509.IPAddress(ipaddress.IPv4Address('127.0.0.1')),
                x509.DNSName(u'localhost'),
                x509.DNSName(ip_str)
            ]),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        )
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
    )

    cert = builder.sign(key, hashes.SHA256())

    with open(key_path, 'wb') as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    with open(cert_path, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return cert_path, key_path

if __name__ == "__main__":
    current_ip = get_current_ip()
    print("=" * 65)
    print(f"📡 الشبكة المتصلة حالياً: IP = {current_ip}")
    print("🔐 جاري تهيئة شهادة الأمان المتوافقة مع الآيفون...")
    cert_file, key_file = generate_ssl_for_ip(current_ip)
    
    scanner_url = f"https://{current_ip}:8443/scanner"
    
    print("=" * 65)
    print("🚀 السيرفر جاهز للعمل!")
    print(f"\n👉 افتح هذا الرابط في Safari على الآيفون:\n\n    {scanner_url}\n")
    print("=" * 65)
    print("💡 في سفاري: اضغط Show Details -> visit this website -> Allow Camera")
    print("=" * 65 + "\n")

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8443, ssl_certfile=cert_file, ssl_keyfile=key_file)
