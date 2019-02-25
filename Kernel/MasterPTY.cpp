#include "MasterPTY.h"
#include "SlavePTY.h"
#include "PTYMultiplexer.h"
#include <Kernel/Process.h>
#include <LibC/errno_numbers.h>
#include <LibC/signal_numbers.h>
#include <LibC/sys/ioctl_numbers.h>

MasterPTY::MasterPTY(unsigned index)
    : CharacterDevice(10, index)
    , m_slave(adopt(*new SlavePTY(*this, index)))
    , m_index(index)
{
    set_uid(current->uid());
    set_gid(current->gid());
}

MasterPTY::~MasterPTY()
{
    dbgprintf("~MasterPTY(%u)\n", m_index);
    PTYMultiplexer::the().notify_master_destroyed(Badge<MasterPTY>(), m_index);
}

String MasterPTY::pts_name() const
{
    return String::format("/dev/pts/%u", m_index);
}

ssize_t MasterPTY::read(Process&, byte* buffer, ssize_t size)
{
    if (!m_slave && m_buffer.is_empty())
        return 0;
    return m_buffer.read(buffer, size);
}

ssize_t MasterPTY::write(Process&, const byte* buffer, ssize_t size)
{
    if (!m_slave)
        return -EIO;
    m_slave->on_master_write(buffer, size);
    return size;
}

bool MasterPTY::can_read(Process&) const
{
    if (!m_slave)
        return true;
    return !m_buffer.is_empty();
}

bool MasterPTY::can_write(Process&) const
{
    return true;
}

void MasterPTY::notify_slave_closed(Badge<SlavePTY>)
{
    dbgprintf("MasterPTY(%u): slave closed, my retains: %u, slave retains: %u\n", m_index, retain_count(), m_slave->retain_count());
    // +1 retain for my MasterPTY::m_slave
    // +1 retain for FileDescriptor::m_device
    if (m_slave->retain_count() == 2)
        m_slave = nullptr;
}

ssize_t MasterPTY::on_slave_write(const byte* data, ssize_t size)
{
    if (m_closed)
        return -EIO;
    m_buffer.write(data, size);
    return size;
}

bool MasterPTY::can_write_from_slave() const
{
    if (m_closed)
        return true;
    return m_buffer.bytes_in_write_buffer() < 4096;
}

void MasterPTY::close()
{
    if (retain_count() == 2) {
        InterruptDisabler disabler;
        // After the closing FileDescriptor dies, slave is the only thing keeping me alive.
        // From this point, let's consider ourselves closed.
        m_closed = true;

        m_slave->hang_up();
    }
}

int MasterPTY::ioctl(Process& process, unsigned request, unsigned arg)
{
    if (request == TIOCSWINSZ)
        return m_slave->ioctl(process, request, arg);
    return -EINVAL;
}
