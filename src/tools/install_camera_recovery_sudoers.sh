#!/bin/bash
set -euo pipefail

TARGET_USER="${1:-robot96}"
SUDOERS_FILE="/etc/sudoers.d/daol-vision-camera-recovery"
SYSTEMCTL_BIN="$(command -v systemctl)"

if [ "$(id -u)" -ne 0 ]; then
    echo "sudo로 실행하세요: sudo $0 ${TARGET_USER}"
    exit 1
fi

if ! id "${TARGET_USER}" >/dev/null 2>&1; then
    echo "사용자 없음: ${TARGET_USER}"
    exit 1
fi

cat > "${SUDOERS_FILE}" <<EOF
# Daol Vision: permit only the Argus daemon commands needed by camera recovery.
${TARGET_USER} ALL=(root) NOPASSWD: ${SYSTEMCTL_BIN} restart nvargus-daemon
${TARGET_USER} ALL=(root) NOPASSWD: ${SYSTEMCTL_BIN} is-active nvargus-daemon
EOF

chmod 0440 "${SUDOERS_FILE}"
visudo -cf "${SUDOERS_FILE}"

set +e
sudo -u "${TARGET_USER}" sudo -n "${SYSTEMCTL_BIN}" is-active nvargus-daemon >/dev/null
CHECK_RC=$?
set -e
if [ "${CHECK_RC}" -ne 0 ] && [ "${CHECK_RC}" -ne 3 ]; then
    echo "비밀번호 없는 systemctl 권한 확인 실패: rc=${CHECK_RC}"
    exit 1
fi

echo "설치 완료: ${SUDOERS_FILE}"
echo "허용 명령: ${SYSTEMCTL_BIN} restart nvargus-daemon"
